
"""
Inference script for the fine-tuned Gemma 2B model for Japanese test case generation.
This script loads a fine-tuned model and generates test cases from Japanese user stories.
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "gemma-2b-japanese-testcase-gen"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate test cases from Japanese user stories")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Japanese user story input"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    
    return parser.parse_args()

def load_model(model_path):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    
    peft_config = PeftConfig.from_pretrained(model_path)
    
    logger.info(f"Loading base model {peft_config.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32,  # Use float32 for CPU inference
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def generate_test_cases(model, tokenizer, user_story, args):
    """
    Generate test cases from a Japanese user story.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        user_story: Japanese user story input
        args: Command line arguments
        
    Returns:
        Generated test cases
    """
    prompt = f"ユーザーストーリー: {user_story}\nテストケース:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    test_cases = generated_text.replace(prompt, "").strip()
    
    return test_cases

def main():
    """Main function to generate test cases."""
    args = parse_args()
    
    model, tokenizer = load_model(args.model_path)
    
    logger.info("Generating test cases")
    test_cases = generate_test_cases(model, tokenizer, args.input, args)
    
    print("\n===== 生成されたテストケース =====")
    print(test_cases)
    print("=================================\n")

if __name__ == "__main__":
    main()
