# TinySQL: Fine-Tuning Lightweight Language Models for BigQuery SQL Generation 

**AAI-590: Applied Artificial Intelligence Capstone Project**  
**University of San Diego**  
**Spring 2026**

**Project Status:** Completed

---

## Team Members

- **Santa Das Karmakar** – sdaskarmakar@sandiego.edu  
- **Sethuraman Krishnasamy** – skrishnasamy@sandiego.edu  
- **Sanjay Kumar** – sanjaykumar@sandiego.edu  

## Instructor

- **Professor Haisav Chokshi**

---

## Project Objective

The goal of this project is to develop **TinySQL** — a lightweight, domain-specific, sub-7B parameter language model optimized for generating accurate **Google BigQuery SQL** from natural language questions.

This project bridges the "Small Model Gap" by fine-tuning **Gemma-2-2b-it** exclusively on BigQuery dialect using high-density synthetic data, **4-bit QLoRA**, and **Flash Attention 2**. The model is designed to run efficiently in serverless environments (GCP Cloud Functions) while ensuring **data sovereignty**.

**Key Objectives:**
- High SQL generation accuracy with small memory footprint
- Full data sovereignty (schema stays within enterprise perimeter)
- Sub-50ms per-token latency for real-time productivity

---

## Methods Used

- Text-to-SQL Generation
- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT / LoRA)
- 4-bit Quantization (QLoRA)
- Flash Attention 2
- BigQuery Dialect Transpilation (SQLGlot)

---

## Technologies

- Python
- Hugging Face Transformers & PEFT
- TRL (SFTTrainer)
- PyTorch
- SQLGlot
- Google Cloud Platform (GCP) – GKE, BigQuery
- Docker & Kubernetes
- Gemma-2-2b-it (base model)

---

## Dataset

- **Source**: [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Training samples: 20,000 (subset)
- Test samples: 1,000

Each example includes: natural language question, database schema, gold SQL, explanation, complexity label, and task type.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/santa-dask/aai-590-tinysql.git
cd aai-590-tinysql
```

### 2. Install Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate bitsandbytes flash-attn
pip install sqlglot pyyaml python-dotenv pandas numpy google-cloud-bigquery google-cloud-storage
```

### 3. Set Environment Variables

Create a `.env` file in the root:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
HF_TOKEN=your-huggingface-token
GCS_BUCKET=your-gcs-bucket-name
```

---

## Quick Start

Run the main notebook:

```bash
jupyter notebook notebook/AAI_590_IN1_Capstone_Project.ipynb
```

Or run inference directly:

```bash
python src/model_training/pretrained_inference.py
```

---

## Repository Structure

```
aai-590-tinysql/
├── README.md
├── Dockerfile
├── config/config.yaml
├── notebook/AAI_590_IN1_Capstone_Project.ipynb
├── job/gemma_job_template.yaml
├── src/
│   ├── model_training/
│   └── utils/
└── setup/
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Professor Haisav Chokshi** – for guidance and support throughout the capstone project
- **University of San Diego – Applied Artificial Intelligence Program**
- Hugging Face, Gretel AI, and the Gemma team for open models and datasets
- Google Cloud for GKE and BigQuery infrastructure support
