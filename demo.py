#!/usr/bin/env python3
# CLI demo that checks prompts using Distilguard and forwards allowed prompts to DeepSeek.
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

merged_model = "models/prompt_shield_merged"
MAX_LENGTH = 512
THRESHOLD = 0.5

# ANSI color codes for CLI 
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
PINK = "\033[95m"  
BOLD = "\033[1m"
RESET = "\033[0m"

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model_and_tokenizer():
    device = get_device()

    # Load tokenizer and merged model from local files 
    try:
        tokenizer = AutoTokenizer.from_pretrained(merged_model, use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            merged_model, num_labels=2, local_files_only=True
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load model/tokenizer from local files. "
            f"Original error: {e}"
        )

    model.to(device).eval()
    return tokenizer, model, device

def classify_text(tokenizer, model, device, text, max_length=MAX_LENGTH):
    # Tokenises input text
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model logits and convert to probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Return probabilities for [injection, safe]
    return float(probs[1]), float(probs[0])

def run_cli(tokenizer, model, device, deepseek_tokenizer=None, deepseek_model=None):
    banner = "**** Prompt Shield Distilguard ****"
    print("\n" + PINK + BOLD + banner + RESET)
    print(PINK + "=" * len(banner) + RESET)
    print("Type a prompt (max 512 tokens), or 'quit' to exit.\n")

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not prompt or prompt.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return

        # Show token count and indicate if input will be truncated to MAX_LENGTH
        try:
            tokenized = tokenizer(prompt)
            token_len = len(tokenized.get("input_ids", []))
        except Exception:
            token_len = None

        if token_len is not None and token_len > MAX_LENGTH:
            print(CYAN + f"[Input tokens: {token_len}] → truncated to {MAX_LENGTH} tokens" + RESET)

        # Classify prompt
        p_inj, p_safe = classify_text(tokenizer, model, device, prompt, max_length=MAX_LENGTH)
        print(f"p_injection={p_inj:.4f}  p_safe={p_safe:.4f}")

        if p_inj >= THRESHOLD:
            print(RED + BOLD + "🚫 BLOCKED: Prompt flagged as injection." + RESET)
        else:
            print(GREEN + BOLD + "✅ ALLOWED: Request forwarded to DeepSeek." + RESET)

            if deepseek_tokenizer is None or deepseek_model is None:
                print(YELLOW + "(DeepSeek generation unavailable)" + RESET)
            else:
                try:
                    # Instruct the model to provide only a concise answer and no internal reasoning
                    messages = [
                        {"role": "user", "content": prompt},
                    ]
                    inputs = deepseek_tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(deepseek_model.device) for k, v in inputs.items()}

                    outputs = deepseek_model.generate(**inputs, max_new_tokens=1024)
                    reply = deepseek_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                    print(PINK + "→ DeepSeek answer:" + RESET, reply)
                except Exception:
                    print(YELLOW + "(DeepSeek generation failed)" + RESET)

def main():
    try:
        tokenizer, model, device = load_model_and_tokenizer()
    except Exception as e:
        print("Failed to load model/tokenizer:", e, file=sys.stderr)
        sys.exit(1)
    
    # Try to load local DeepSeek once at startup 
    deepseek_tokenizer = None
    deepseek_model = None
    deepseek_path = "models/deepseek"
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_path, use_fast=True, local_files_only=True)
        deepseek_model = AutoModelForCausalLM.from_pretrained(deepseek_path, local_files_only=True)
        deepseek_model.to(device).eval()
        print(PINK + "Loaded DeepSeek model from:" + RESET, deepseek_path)
    except Exception:
        deepseek_tokenizer = None
        deepseek_model = None

    run_cli(tokenizer, model, device, deepseek_tokenizer, deepseek_model)

if __name__ == "__main__":
    main()
