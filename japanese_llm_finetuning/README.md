# Japanese LLM Fine-tuning for Test Case Generation

This project fine-tunes Google's Gemma 2B model to convert Japanese user stories into test cases.

## Project Structure

- `data/`: Contains training data in JSONL format
- `models/`: Directory for storing fine-tuned models
- `scripts/`: Python scripts for fine-tuning and inference

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning

```bash
python scripts/finetune.py
```

### Inference

```bash
python scripts/inference.py --input "ユーザーストーリー"
```

## Model Details

- Base model: Google's Gemma 2B
- Fine-tuning technique: LoRA (Low-Rank Adaptation) with PEFT (Parameter-Efficient Fine-Tuning)
- Optimization: CPU-only for MacBook Air M1
