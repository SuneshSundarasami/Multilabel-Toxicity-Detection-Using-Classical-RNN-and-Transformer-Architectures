import pandas as pd
from datasets import Dataset

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_and_check_data(path, logger):
    train_data = pd.read_csv(path)
    required_columns = ['comment_text'] + LABEL_COLS
    missing = [col for col in required_columns if col not in train_data.columns]
    if missing:
        logger.error(f"Missing columns in dataset: {missing}")
        raise ValueError(f"Missing columns in dataset: {missing}")
    logger.info("All required columns found in dataset")
    return train_data

def preprocess_data(train_data, logger):
    train_data['comment_text'] = train_data['comment_text'].fillna("")
    for col in LABEL_COLS:
        train_data[col] = train_data[col].fillna(0).astype(int)
    logger.info("Label distribution:")
    for col in LABEL_COLS:
        count = train_data[col].sum()
        percentage = (count / len(train_data)) * 100
        logger.info(f"  {col}: {count} ({percentage:.2f}%)")
    train_data['input'] = train_data.apply(lambda row: ', '.join([f"{col}={int(row[col])}" for col in LABEL_COLS]), axis=1)
    train_data['output'] = train_data['comment_text']
    train_data['text'] = "<toxicity> " + train_data['input'] + " </toxicity> <comment> " + train_data['output']
    return train_data

def get_hf_dataset(train_data, development_mode=True, dev_size=5000):
    if development_mode:
        train_data = train_data.head(dev_size)
    return Dataset.from_pandas(train_data[['text']])