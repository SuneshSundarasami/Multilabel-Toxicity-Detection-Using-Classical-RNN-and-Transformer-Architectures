# Model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Data
TRAIN_DATA_PATH = "/home/sunesh/NLP/Multi_Label_Toxic_Comment_Classifier/Dataset/train_preprocessed.csv"
# TRAIN_DATA_PATH = "/work/ssunda2s/toxic_comment_dataset/train_preprocessed.csv"
# Output directories
OUTPUT_DIR = "./mistral7b-toxic"
FINAL_MODEL_DIR = "./mistral7b-toxic-final"
LOGS_DIR = "./logs"
CACHE_DIR = "./cache"

# Training summary
def get_summary_file():
    from datetime import datetime
    return f"{LOGS_DIR}/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"