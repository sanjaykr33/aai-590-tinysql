import json
import os
import torch
import re
import sqlglot
from sqlglot import exp
from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
file_path = os.environ.get("FILE_PATH")
output_log_path = os.environ.get("OUTPUT_LOG_PATH")
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
FULL_DATASET_PATH = f"{PROJECT_ID}.{DATASET_ID}"
LOCATION = os.environ.get("LOCATION")
MODEL_ID = os.environ.get("MODEL_ID")

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
    text = re.split(r'(?i)note:|explanation:|---|\n\n', text)[0].strip()

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
        return False, str(e)

def extract_and_fix_ddl(context_raw):
    """Transpiles MySQL DDL to BigQuery and strips constraints."""
    try:
        statements = sqlglot.parse(context_raw, read="mysql")
    except:
        return ""

    ddl_statements = []
    for expression in statements:
        if isinstance(expression, exp.Create) and expression.args.get("kind") == "TABLE":
            expression.set("exists", True)

            if isinstance(expression.this, exp.Schema):
                table_ident = expression.this.this
            else:
                table_ident = expression.this
            if isinstance(table_ident, exp.Table):
                table_ident.set("db", None)
                table_ident.set("catalog", None)

            schema = expression.this
            if isinstance(schema, exp.Schema):
                for column_def in schema.expressions:
                    if isinstance(column_def, exp.ColumnDef):
                        column_def.set("constraints", [
                            c for c in column_def.args.get("constraints", [])
                            if not (isinstance(c.kind, (exp.PrimaryKeyColumnConstraint, exp.Reference)) or "FOREIGN" in str(c.kind).upper())
                        ])
                schema.set("expressions", [
                    e for e in schema.expressions
                    if not (isinstance(e, exp.Constraint) and any(isinstance(k, (exp.PrimaryKey, exp.ForeignKey, exp.Reference)) for k in e.args.values()))
                ])
            ddl_statements.append(expression.sql(dialect="bigquery"))
    return ";\n".join(ddl_statements)

def generate_sql(prompt, schema):
    """Generates SQL with hard-coded dataset anchoring and a code trigger."""
    input_text = f"""<start_of_turn>user
You are a GoogleSQL expert. Generate a BigQuery query to answer the question using the schema below.
Rules:
1. Use ONLY the table and columns provided in the schema.
2. Prefix all table names with `{DATASET_ID}.`.
3. Return ONLY the SQL query.

Schema:
{schema}

Question:
{prompt}<end_of_turn>
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
                is_valid, bq_msg = validate_sql_dry_run(gen_sql)

                res = {
                    "prompt": sql_prompt,
                    "original_sql": gold_sql,
                    "generated_sql": gen_sql,
                    "valid": is_valid
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