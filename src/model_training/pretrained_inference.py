import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from google.cloud import bigquery


#Processing record: 3 , 13, 24, 37,133,

from utils.sql_utils import validate_sql_dry_run
from utils import sql_utils


PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
GCS_BUCKET = os.environ.get("GCS_BUCKET")


FULL_DATASET_PATH = f"{PROJECT_ID}.{DATASET_ID}"
file_path = f"/gcs/{GCS_BUCKET}/data/tinycode_test_ds.jsonl"
output_file_path=f"/gcs/{GCS_BUCKET}/data/pretrained_inference_results.jsonl"
stop_markers = ["Explanation:", "Result:", "Expected Result", "Note:", "\n\n"]


def initialize_hf_model():
    model_path = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically places model on GPU if available
        torch_dtype=torch.bfloat16
    )
    return model, tokenizer

def initialize_pretrained_model(type="pretrained"):
    # 1. Path to your fine-tuned model directory (or adapter weights if using LoRA)
    model_path = "/app/gemma-2-2b-sql-finetuned"

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically places model on GPU if available
        torch_dtype=torch.bfloat16
    )
    return model, tokenizer

def initialize_model(type="pretrained"):
  if type == "pretrained":
    return initialize_pretrained_model()
  elif type == "hf":
    return initialize_hf_model()
  else:
    raise ValueError("Invalid model type")

def generate_sql(question, table_schema, model, tokenizer):
    # 4. Construct the prompt dynamically using the arguments
    sql_prompt = f"""Write a BigQuery SQL query to answer the user's question.
        Schema:
        {table_schema}

        Question:
        {question}

        SQL Query:
        """
    #print(sql_prompt)
    inputs = tokenizer(sql_prompt, return_tensors="pt").to(model.device)

    # Adjust max_new_tokens based on how long you expect the SQL query to be
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.1,  # Keep temperature low for code/SQL generation
        do_sample=True,
    )

    # 5. Decode the output
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print just the generated SQL (stripping out the original prompt)
    bq_sql = prediction[len(sql_prompt) :].strip()
    #print(sql_query)
    return bq_sql


def extract_and_fix_ddl(context_raw):
    """Transpiles MySQL DDL to BigQuery and strips constraints."""
    return sql_utils.extract_and_fix_ddl(context_raw)


def get_sql_results(sql_prompt, sql_context, model, tokenizer):
              # Generate SQL
            bq_sql = generate_sql(sql_prompt, sql_context, model, tokenizer)

            # Truncate and patch as before
            for marker in stop_markers:
                bq_sql = bq_sql.split(marker)[0]
            bq_sql = bq_sql.strip()
            bq_sql = re.sub(r"\b\d{20,}\b", "0", bq_sql)

            # Replace placeholder and Validate
            sql_query = bq_sql.replace(
                "{DATASET_ID}", f"{PROJECT_ID}.{DATASET_ID}"
            )
            sql_query=sql_query.lower()

            print(f"Validating SQL:\n{sql_query}")
            is_valid, message = validate_sql_dry_run(sql_query)
            return bq_sql, is_valid, message

def create_table_if_not_exists(ddl, count):
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(default_dataset=FULL_DATASET_PATH)
    try:
        client.query(ddl, job_config=job_config).result()
        return True
    except Exception as e:
        print(f"DDL Setup Error Record {count}: {e}")
        return False

def process_records(file_path):

    with open(file_path, "r", encoding="utf-8") as f, open(
        output_file_path, "w", encoding="utf-8"
    ) as out_f:

        for count, line in enumerate(f):
            print(f"Processing record: {count}")
            if count > 200:
                break
            record = json.loads(line)
            sql_prompt = record.get("sql_prompt", "")
            sql_context = record.get("sql_context", "")  # Contains DDL

            final_ddl = extract_and_fix_ddl(record.get("sql_context", ""))
            final_ddl = final_ddl.lower()

            table_exists = create_table_if_not_exists(final_ddl, count)
            if not table_exists:
              pass

            bq_sql, is_valid, message = get_sql_results(sql_prompt, sql_context, model, tokenizer)

            # Prepare the result dictionary and write to file
            result_data = {
                "record_index": count,
                "prompt": sql_prompt,
                "generated_sql": bq_sql,
                "dry_run_valid": is_valid,
                "failure_reason": message if not is_valid else "Success",
            }
            out_f.write(json.dumps(result_data) + "\n")

            print(f"Dry-run Valid: {is_valid}")
            print(f"Dry-run Message: {message}")
            print("-" * 50)

def individual_record_test(custom_sql=False):
    if not custom_sql:
        sql_prompt = """Find the average travel time (in days) for vessels traveling
                         between the Port of Valparaiso and the Port of Singapore in
                         the first half of 2021, ranked by the average travel time in
                         ascending order."""

        sql_context = """
        CREATE TABLE Routes (route_id INT, departure_port VARCHAR(20), arrival_port VARCHAR(20));
        CREATE TABLE VesselTravel (vessel_id INT, route INT, departure_date DATE, travel_time INT);
        INSERT INTO Routes (route_id, departure_port, arrival_port) VALUES
        (1, 'Los Angeles', 'Tokyo'), (2, 'Rotterdam', 'New York'), (3, 'Santos', 'Hong Kong'),
        (4, 'Mumbai', 'Shanghai'), (5, 'Buenos Aires', 'Jakarta'), (6, 'Dakar', 'Lagos'),
        (7, 'Valparaiso', 'Singapore');
        INSERT INTO VesselTravel (vessel_id, route, departure_date, travel_time) VALUES
        (1, 7, '2021-01-01', 45), (2, 7, '2021-02-01', 46), (3, 7, '2021-03-01', 47),
        (4, 7, '2021-04-01', 44), (5, 7, '2021-05-01', 45), (6, 7, '2021-06-01', 46);
        """
        final_ddl = extract_and_fix_ddl(sql_context)
        final_ddl = final_ddl.lower()

        table_exists = create_table_if_not_exists(final_ddl, 1)
        if not table_exists:
            pass
            bq_sql, is_valid, message = get_sql_results(sql_prompt, sql_context, model, tokenizer)
            print (bq_sql, is_valid)



if __name__ == "__main__":
    model, tokenizer = initialize_model()
    process_records(file_path)
    #individual_record_test(custom_sql=True)