from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AutoModel
import os

app = Flask(__name__)

# Copy your model class from the notebook
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, model_name="bert-base-uncased", dropout_rate=0.1):
        super(BertForMultiLabelClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if 'distilbert' in self.bert.config._name_or_path or 'roberta' in self.bert.config._name_or_path:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

# Simple predictor class
class ToxicPredictor:
    def __init__(self, model_path, model_type):
        self.device = torch.device('cpu')
        self.model_type = model_type
        
        # Load tokenizer
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMultiLabelClassification(6, 'bert-base-uncased')
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = BertForMultiLabelClassification(6, 'roberta-base')
        else:  # distilbert
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = BertForMultiLabelClassification(6, 'distilbert-base-uncased')
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def predict(self, text):
        inputs = self.tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            if self.model_type == 'bert':
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
            else:
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        probs = outputs.squeeze().numpy()
        return {
            'is_toxic': bool(np.any(probs > 0.5)),
            'probabilities': {label: float(prob) for label, prob in zip(self.labels, probs)}
        }

# Load models
models = {
    'BERT': ToxicPredictor('models/bert_toxic_classifier.pt', 'bert'),
    'RoBERTa': ToxicPredictor('models/roberta_toxic_classifier.pt', 'roberta'),
    'DistilBERT': ToxicPredictor('models/distilbert_toxic_classifier.pt', 'distilbert')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model', 'BERT')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = models[model_name].predict(text)
        return jsonify({'model': model_name, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use environment port for cloud deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)