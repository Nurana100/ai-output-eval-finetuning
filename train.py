"""
Fine-tunes google/flan-t5-small on train_data.jsonl to correct the
"answers confidently instead of declining on out-of-scope questions"
failure category.

RUN THIS IN GOOGLE COLAB (not in a plain local machine unless you have a
GPU) — this sandbox environment can't reach huggingface.co to download the
base model, so this script is written to be copy-pasted into a Colab
notebook where you have full internet access and a free GPU:

  1. Go to colab.research.google.com -> New notebook
  2. Runtime > Change runtime type > T4 GPU
  3. First cell: !pip install -q transformers datasets accelerate
  4. Upload finetune/train_data.jsonl (Files panel, drag and drop)
  5. Paste this whole file into a cell and run it
  6. Download the resulting ./flan-t5-nimbus folder, or just keep the
     before/after answers printed at the end for your report

Expect this to take 2-5 minutes on a Colab T4 GPU with this tiny dataset.
"""
import json

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 64


def load_data(path="train_data.jsonl"):
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            rows.append(
                {
                    "input": f"Answer using only this context. If the context doesn't cover it, say you don't have that information.\n\nContext: {ex['context']}\n\nQuestion: {ex['question']}",
                    "target": ex["answer"],
                }
            )
    return Dataset.from_list(rows)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_data()

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"], max_length=MAX_INPUT_LEN, truncation=True
        )
        labels = tokenizer(
            batch["target"], max_length=MAX_TARGET_LEN, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    # --- BEFORE: sample a couple of out-of-scope questions pre-finetune ---
    ood_probe_questions = [
        "Does Nimbus Notes have a built-in calendar or reminders feature?",
        "Can I make video calls through Nimbus Notes?",
    ]

    def generate(q, context=""):
        prompt = f"Answer using only this context. If the context doesn't cover it, say you don't have that information.\n\nContext: {context}\n\nQuestion: {q}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN)
        out = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    print("=== BEFORE fine-tuning ===")
    before = {}
    for q in ood_probe_questions:
        before[q] = generate(q)
        print(f"Q: {q}\nA: {before[q]}\n")

    # --- Fine-tune ---
    args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-nimbus",
        num_train_epochs=8,
        per_device_train_batch_size=4,
        learning_rate=3e-4,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    # --- AFTER: same probe questions, post-finetune ---
    print("\n=== AFTER fine-tuning ===")
    after = {}
    for q in ood_probe_questions:
        after[q] = generate(q)
        print(f"Q: {q}\nA: {after[q]}\n")

    with open("before_after_comparison.json", "w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)

    model.save_pretrained("./flan-t5-nimbus")
    tokenizer.save_pretrained("./flan-t5-nimbus")
    print("Saved model to ./flan-t5-nimbus and comparison to before_after_comparison.json")


if __name__ == "__main__":
    main()
