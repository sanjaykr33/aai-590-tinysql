import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import sql_utils

if "KUBERNETES_SERVICE_HOST" in os.environ:
    del os.environ["KUBERNETES_SERVICE_HOST"]
os.environ["GKE_DIAGON_IDENTIFIER"] = os.environ.get("HOSTNAME", "sft-pod-1")
os.environ["GKE_DIAGON_METADATA"] = '{"project_id":"gpu-launchpad-playground","location":"us-central1","cluster_name":"sft-cluster"}'


import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# --- CONFIGURATION ---
MODEL_ID = os.environ.get("MODEL_ID")
DATASET_ID = os.environ.get("DATASET_ID")
OUTPUT_DIR = "./gemma-2-2b-sql-finetuned"
GCS_BUCKET = os.environ.get("GCS_BUCKET")
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN") # Ensure your token is set for Gemma access


# --- 1. DATA PROCESSING ---
print("Loading and formatting dataset...")

# Load the dataset you mentioned
dataset = load_dataset("gretelai/synthetic_text_to_sql")

def format_instruction(example):
    """
    Formats the data to match your exact inference prompt template.
    We append the target SQL to the end of the model turn.
    """
    prompt = example['sql_prompt']
    schema = sql_utils.extract_and_fix_ddl(example['sql_context'])  # Fix DDL
    target_sql = sql_utils.transpile_to_bigquery(example['sql'])     # Fix Query to BQ
    target_sql = sql_utils.prefix_table_names(target_sql, "{DATASET_ID}") # Lit prefix

    # Notice this exactly mirrors your generate_sql() structure
    text = f"""<start_of_turn>user
    You are a GoogleSQL expert. Generate a BigQuery query to answer the question using the schema below.
    Rules:
    1. Use ONLY the table and columns provided in the schema.
    2. Prefix all table names with `{DATASET_ID}.`.
    3. Return ONLY the SQL query.

    Schema:
      {schema}

    Question:
      {prompt}
    <end_of_turn>
    <start_of_turn>model
    SQL:{target_sql}<end_of_turn>"""

    return {"text": text}

# Apply formatting and shuffle
PROCESSED_DATA_CACHE_DIR = f"/gcs/{GCS_BUCKET}/data/processed_dataset_cache_splits"

if os.path.exists(PROCESSED_DATA_CACHE_DIR):
    print(f"Loading preprocessed dataset splits from cache: {PROCESSED_DATA_CACHE_DIR}")
    dataset_splits = load_from_disk(PROCESSED_DATA_CACHE_DIR)
    train_dataset = dataset_splits["train"]
    val_dataset = dataset_splits["test"]
else:
    print("No cache found. Preprocessing and transpiling records...")
    split_dataset = dataset["train"].train_test_split(test_size=1000, seed=42)
    train_subset = split_dataset["train"].shuffle(seed=42).select(range(20000))
    val_subset = split_dataset["test"]

    train_dataset = train_subset.map(format_instruction)
    val_dataset = val_subset.map(format_instruction)

    from datasets import DatasetDict
    dataset_splits = DatasetDict({"train": train_dataset, "test": val_dataset})

    print(f"Saving preprocessed dataset splits to cache: {PROCESSED_DATA_CACHE_DIR}")
    dataset_splits.save_to_disk(PROCESSED_DATA_CACHE_DIR)

# --- 2. MODEL & TOKENIZER SETUP (QLoRA) ---
print(f"Loading {MODEL_ID} in 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGINGFACE_TOKEN)
tokenizer.padding_side = 'right' # Recommended for Gemma

# 4-bit Quantization Config to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)

model = prepare_model_for_kbit_training(model)

# --- 3. LORA CONFIGURATION ---
# We target the attention modules for adaptation
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    #max_steps=500,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=200,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    #  These MUST live in SFTConfig now, not SFTTrainer
    max_length=1024,
    dataset_text_field="text",
)


# --- 3. INJECT INTO SFTTRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    processing_class=tokenizer
)

# --- 4. EXECUTE TRAINING ---
print("Starting training...")

trainer.train()

# --- 7. SAVE ADAPTERS ---
print(f"Saving fine-tuned model adapters to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")