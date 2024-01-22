from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd

transcripts_file = 'transcripts.csv'
transcripts_df = pd.read_csv(transcripts_file)

transcripts = transcripts_df['transcript'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=200)

tokenized_inputs = tokenizer(transcripts, padding=True, truncation=True, return_tensors="pt")

data_loader = DataLoader(TensorDataset(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]), batch_size=32, shuffle=False)

model.eval()
with torch.no_grad():
    predictions = []
    for batch in data_loader:
        input_ids, attention_mask = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())

transcripts_df['predicted_category'] = predictions

transcripts_df.to_csv('path/to/predicted_transcripts.csv', index=False)
