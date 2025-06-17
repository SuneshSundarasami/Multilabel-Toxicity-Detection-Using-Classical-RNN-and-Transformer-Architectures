import os
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from logger_utils import setup_logging, log_system_info
from data_utils import load_and_check_data, preprocess_data, get_hf_dataset
from tokenizer_utils import get_tokenizer, tokenize_dataset
from custom_callback import CustomLoggingCallback
from config import MODEL_NAME, TRAIN_DATA_PATH, OUTPUT_DIR, FINAL_MODEL_DIR, LOGS_DIR, CACHE_DIR, get_summary_file
from datetime import datetime

def main():
    log_file = setup_logging(LOGS_DIR)
    logger = logging.getLogger(__name__)
    logger.info("Starting toxic comment generator training")
    os.environ["WANDB_DISABLED"] = "true"
    log_system_info(logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_data = load_and_check_data(TRAIN_DATA_PATH, logger)
    train_data = preprocess_data(train_data, logger)
    dataset = get_hf_dataset(train_data, development_mode=True, dev_size=5000)
    logger.info(f"Dataset size: {len(dataset)}")

    # Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    logger.info("Tokenizer initialized successfully")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, num_proc=4)
    logger.info(f"Tokenized dataset: {tokenized_dataset}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        logging_steps=10,
        max_steps=10000,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to=None,
        logging_dir=None,
    )

    logger.info("Training configuration:")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  FP16: {training_args.fp16}")

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model loaded. Total parameters: {model.num_parameters():,}")

    # Trainer
    custom_callback = CustomLoggingCallback(log_file)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[custom_callback],
    )

    # Train
    try:
        logger.info("Starting training...")
        logger.info(f"Log file location: {log_file}")
        trainer.train()
        logger.info("Saving model...")
        model.save_pretrained(FINAL_MODEL_DIR)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to {FINAL_MODEL_DIR}")
        summary_file = get_summary_file()
        summary = {
            "training_completed": True,
            "final_step": trainer.state.global_step,
            "dataset_size": len(dataset),
            "model_parameters": model.num_parameters(),
            "training_args": {
                "batch_size": training_args.per_device_train_batch_size,
                "max_steps": training_args.max_steps,
                "learning_rate": training_args.learning_rate,
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training summary saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()