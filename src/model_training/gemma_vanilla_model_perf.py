import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import sql_utils
import json
import os
import torch
import re
try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
GCS_BUCKET = os.environ.get("GCS_BUCKET")

file_path = f'/gcs/{GCS_BUCKET}/data/tinycode_test_ds.jsonl'
output_log_path = f'/gcs/{GCS_BUCKET}/data/gemma_benchmark_results.jsonl'

FULL_DATASET_PATH = f"{PROJECT_ID}.{DATASET_ID}"
LOCATION = "us-central1"
MODEL_ID = "google/gemma-2-2b"

# Initialize BigQuery Client
client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

# --- MODEL LOADING ---
print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def setup_bigquery_environment():
    dataset = bigquery.Dataset(FULL_DATASET_PATH)
    dataset.location = LOCATION
    try:
        client.create_dataset(dataset, timeout=30)
    except Conflict:
        pass

def extract_sql_only(text):
    """
    Cleans model chatter and extracts the actual SQL statement.
    Fixed: No longer splits on 'table' or 'schema' which are valid SQL keywords.
    """

    #print (f"Raw Generated SQL : {text}")

    # Remove the trigger prefix if model repeats it
    text = re.sub(r'^(Answer:|SQL:|\s+)', '', text, flags=re.IGNORECASE).strip()

    # Extract everything until the first semicolon
    if ';' in text:
        text = text.split(';')[0].strip() + ';'

    # Remove common hallucinated trailing text
    text = re.split(r'(?i)answer:|note:|explanation:|---|\n\n', text)[0].strip()

    # Final cleanup of non-ASCII garbage
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def validate_sql_dry_run(sql_query):
    """Validates the generated SQL via BigQuery Dry Run."""
    if not sql_query or not any(k in sql_query.upper() for k in ["SELECT", "DELETE", "UPDATE", "INSERT", "WITH"]):
        return False, "Malformed or Empty SQL"

    job_config = bigquery.QueryJobConfig(dry_run=True, default_dataset=FULL_DATASET_PATH)
    try:
        client.query(sql_query, job_config=job_config)
        return True, "Success"
    except Exception as e:
        error_msg = str(e)
        # Strip BigQuery URL
        error_msg = re.sub(r"400 POST https://bigquery\.googleapis\.com/bigquery/v2/projects/[^/]+/jobs\?prettyPrint=false:?", "", error_msg, flags=re.IGNORECASE).strip()
        # De-identify Project ID
        error_msg = re.sub(f"{PROJECT_ID}", "[PROJECT_ID]", error_msg, flags=re.IGNORECASE)
        # Strip Job ID
        error_msg = re.sub(r"Job ID:\s*[a-f0-9\-]+", "", error_msg, flags=re.IGNORECASE)
        # Strip Location
        error_msg = re.sub(r"Location:\s*[a-z0-9\-]+", "", error_msg, flags=re.IGNORECASE)
        # Clean double spaces and leading characters
        error_msg = re.sub(r"\s+", " ", error_msg).strip()
        error_msg = re.sub(r"^:\s*", "", error_msg)
        return False, error_msg

def extract_and_fix_ddl(context_raw):
    """Transpiles MySQL DDL to BigQuery and strips constraints."""
    return sql_utils.extract_and_fix_ddl(context_raw)

def generate_sql(prompt, schema):
    """Generates SQL with hard-coded dataset anchoring and a code trigger."""
    input_text = f"""<start_of_turn>user
    You are a GoogleSQL expert. Generate a BigQuery query to answer the question using the schema below.
    Rules:
    1. Use ONLY columns provided in the schema. If a column is not explicitly defined in the CREATE TABLE statement, you MUST NOT use it. Do not assume standard names exist.
    2. Prefix all table names with `{DATASET_ID}.`.
    3. Return ONLY the SQL query.

    Schema:
    {schema}

    Question:
    {prompt}
    <end_of_turn>
    <start_of_turn>model
    SQL:""" # Triggering the start of the code block

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,        # Use greedy decoding to minimize randomness
            #repetition_penalty=1.2, # Discourage repeating the prompt
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return extract_sql_only(decoded)

def process_records(path, count=500):
    if not os.path.exists(path):
        print(f"Path {path} not found.")
        return

    setup_bigquery_environment()
    results_to_save = []

    print(f"Benchmarking {count} records...")

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= count: break
            record = json.loads(line)

            sql_prompt = record.get("sql_prompt", "")
            gold_sql = record.get("sql", "")
            final_ddl = extract_and_fix_ddl(record.get("sql_context", ""))

            # 1. Execute DDL so the table exists for validation
            job_config = bigquery.QueryJobConfig(default_dataset=FULL_DATASET_PATH)
            table_exists = False
            if final_ddl:
                try:
                    client.query(final_ddl, job_config=job_config).result()
                    table_exists = True
                except Exception as e:
                    print(f"DDL Setup Error Record {i+1}: {e}")

            # 2. Generate and Validate SQL
            if table_exists:
                gen_sql = generate_sql(sql_prompt, final_ddl)

                # Attempt to transpile from MySQL to BigQuery (fallback for MySQL-generated syntax)
                try:
                    gen_sql = sql_utils.transpile_to_bigquery(gen_sql)
                except Exception:
                    # Fallback to original if transpilation fails
                    pass

                # Swap literal placeholder for actual BigQuery dataset before validation
                run_sql = gen_sql.replace("{DATASET_ID}", DATASET_ID)
                is_valid, bq_msg = validate_sql_dry_run(run_sql)

                res = {
                    "prompt": sql_prompt,
                    "original_sql": gold_sql,
                    "generated_sql": gen_sql,
                    "valid": is_valid,
                    #"error": bq_msg if not is_valid else None
                }
                results_to_save.append(res)

                print(f"--- Record {i+1} ---")
                print(f"Q: {sql_prompt}")
                print(f"Gold SQL : {gold_sql}")
                print(f"GEMMA SQL: {gen_sql}")
                print(f"DRY RUN: {'PASS' if is_valid else 'FAIL'}")
                if not is_valid: print(f"REASON: {bq_msg}")
                print("-" * 30)
            else:
                print(f"Skipping Record {i+1} due to missing schema context.")

    with open(output_log_path, 'w') as out_f:
        for entry in results_to_save:
            out_f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    process_records(file_path)