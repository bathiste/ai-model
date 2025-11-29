import sqlite3
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

DB = "dataset.db"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fits on 12GB VRAM

# --------------------------------------------------
# LOAD TEXT FROM SQLITE
# --------------------------------------------------
conn = sqlite3.connect(DB)
rows = conn.execute("SELECT text_content FROM documents").fetchall()
conn.close()

texts = [r[0] for r in rows if r[0] and len(r[0]) > 50]

print("Loaded texts:", len(texts))

dataset = Dataset.from_dict({"text": texts})

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# --------------------------------------------------
# MODEL + LoRA
# --------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_cfg)

# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------
args = TrainingArguments(
    output_dir="trained-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=1500,      # SAFE for 3060
    fp16=False,
    bf16=True,
    logging_steps=20,
    save_steps=300,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("trained-model")
tokenizer.save_pretrained("trained-model")
