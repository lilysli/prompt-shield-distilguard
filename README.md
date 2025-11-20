## 📁 Project Overview

This repository demonstrates a small **prompt-shielding pipeline** using a classifier to filter unsafe prompts before forwarding them to a local LLM.

---

### **1. `prompt_shield_pipeline.ipynb` — Classifier Training & Evaluation**

Trains and evaluates the DistilGuard classifier using:

- **Model:** DistilBERT (`distilbert/distilbert-base-uncased`)  
 https://huggingface.co/distilbert/distilbert-base-uncased  
- **Dataset:** Safe-Guard Prompt Injection by **xTRam1**  
 https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection

The notebook handles dataset loading, fine-tuning, evaluation, and saving the resulting classifier.

---

### **2. `download_deepseek_model.ipynb` — Local DeepSeek Model Download**

Downloads and stores the **DeepSeek-R1-Distill-Qwen-1.5B** model locally:
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

This prepares the model for offline use in the CLI demo.

---

### **3. `demo.py` — CLI Prompt Shield Demo**

A simple command-line interface that:

1. Takes user input  
2. Runs it through the trained classifier  
3. If **safe**, forwards it to the local DeepSeek model  
4. If **unsafe**, blocks the request

This provides a minimal example of using a classifier as a guardrail in front of an LLM.


![Screenshot of CLI demo](Demo.png)
