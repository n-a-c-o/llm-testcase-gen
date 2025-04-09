
"""
Fine-tune Google's Gemma 2B model using LoRA and PEFT for Japanese test case generation.
This script is optimized for CPU-only environments like MacBook Air M1.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_NAME = "google/gemma-2b"
DEFAULT_TRAINING_DATA = DATA_DIR / "training_data.jsonl"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "gemma-2b-japanese-testcase-gen"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2B for Japanese test case generation")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--training_data", 
        type=str, 
        default=str(DEFAULT_TRAINING_DATA),
        help="Path to training data in JSONL format"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=1,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=8,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout probability"
    )
    
    return parser.parse_args()

def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSONL file and convert to HuggingFace Dataset.
    
    Args:
        data_path: Path to JSONL file with training data
        
    Returns:
        HuggingFace Dataset
    """
    logger.info(f"Loading training data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    dataset = Dataset.from_pandas(df)
    
    logger.info(f"Loaded {len(dataset)} training examples")
    return dataset

def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess training examples by formatting and tokenizing them.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer for the model
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    prompts = []
    for input_text, output_text in zip(examples["input"], examples["output"]):
        prompt = f"ユーザーストーリー: {input_text}\nテストケース: {output_text}"
        prompts.append(prompt)
    
    tokenized_examples = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    tokenized_examples["labels"] = tokenized_examples["input_ids"].clone()
    
    return tokenized_examples

def create_peft_config(args):
    """
    Create LoRA configuration for PEFT.
    
    Args:
        args: Command line arguments
        
    Returns:
        LoRA configuration
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )

def main():
    """Main function to fine-tune the model."""
    args = parse_args()
    
    set_seed(args.seed)
    
    dataset = load_training_data(args.training_data)
    
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Preprocessing dataset")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"Loading base model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU training
        device_map="auto",
    )
    
    logger.info("Preparing model for LoRA fine-tuning")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Creating PEFT model with LoRA config")
    peft_config = create_peft_config(args)
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        fp16=False,  # Disable mixed precision for CPU training
        report_to="none",  # Disable wandb, tensorboard, etc.
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()
