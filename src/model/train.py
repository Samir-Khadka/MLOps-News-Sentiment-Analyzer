import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../../models/distilbert-finance")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("Loading dataset...")
    # Load financial_phrasebank as a proxy for FiQA/TweetEval for simplicity
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    
    # Split dataset
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3 labels: positive, negative, neutral
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    if eval_results['eval_accuracy'] >= 0.70:
        print(f"Model meets accuracy requirement: {eval_results['eval_accuracy']} >= 0.70")
    else:
        print(f"Warning: Model accuracy {eval_results['eval_accuracy']} is below 0.70 target")
        
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
