import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import os

def main():
    # Check for GPU
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load dataset
    train_data = pd.read_csv('/home/sunesh/NLP/Multi_Label_Toxic_Comment_Classifier/Dataset/train_preprocessed.csv')
    print(f"Dataset shape: {train_data.shape}")
    print(f"Sample data:\n{train_data.head(2)}")

    # Check if any required columns are missing
    required_columns = ['comment_text'] + ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    missing_columns = [col for col in required_columns if col not in train_data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    # Format data
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    def format_labels(row):
        return ', '.join([f"{col}={int(row[col])}" for col in label_cols])

    # Handle potential NaNs
    train_data['comment_text'] = train_data['comment_text'].fillna("")
    for col in label_cols:
        train_data[col] = train_data[col].fillna(0).astype(int)

    # Create formatted text
    train_data['input'] = train_data.apply(format_labels, axis=1)
    train_data['output'] = train_data['comment_text']
    train_data['text'] = "<toxicity> " + train_data['input'] + " </toxicity> <comment> " + train_data['output']

    # Convert to dataset
    dataset = Dataset.from_pandas(train_data[['text']])
    print(f"Created dataset with {len(dataset)} examples")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Tokenize inputs
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
        # Set up labels for language modeling (same as input_ids)
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    print(f"Tokenized dataset: {tokenized_dataset}")

    # Configure training
    training_args = TrainingArguments(
        output_dir="./gpt2-toxic",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        logging_steps=10,
        max_steps=10,
        save_steps=5,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    # Load model and configure trainer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    try:
        print("Starting training...")
        trainer.train()
        
        # Save model
        model.save_pretrained("./gpt2-toxic-final")
        tokenizer.save_pretrained("./gpt2-toxic-final")
        print("Training completed successfully!")
        print("Model saved to ./gpt2-toxic-final")
        
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    main()