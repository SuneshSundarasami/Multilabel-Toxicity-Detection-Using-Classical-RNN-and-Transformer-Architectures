import pandas as pd
import pickle
import random
import yaml
import os

# Load config
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'locations.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Load processed data
train_path = config['data']['train_tokens']  # Direct path now
with open(train_path, 'rb') as f:
    data = pickle.load(f)

tokens = data['tokens']
labels = data['labels']

# Load original data
train_data = pd.read_csv(config['data']['train_raw'])
toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("🔤 TOKENIZED EXAMPLES")
print("=" * 50)

# Show 5 random toxic examples as tokens
random.seed(42)
toxic_indices = [i for i, row in enumerate(labels) if sum(row) > 0]
sample_indices = random.sample(toxic_indices, 5)

for i, idx in enumerate(sample_indices, 1):
    original = train_data.iloc[idx]['comment_text']
    token_list = tokens[idx]
    toxic_labels = [toxic_columns[j] for j, val in enumerate(labels[idx]) if val == 1]
    
    print(f"\nExample {i} - Labels: {', '.join(toxic_labels)}")
    print(f"Original: {original[:80]}...")
    print(f"Tokens:   {token_list}")