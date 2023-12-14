from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from CSV
import pandas as pd

df = pd.read_csv('archive/Tweets.csv')
df.head()

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels for negative, positive, neutral

# Define your dataset class
class TweetDataset(Dataset):
    def __init__(self, tweets, labels):
        self.tweets = tweets
        self.labels = labels

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return {'text': self.tweets[idx], 'label': self.labels[idx]}

# Tokenize and create DataLoader for training set
train_tweets = train_df['selected_text'].tolist()
train_labels = train_df['sentiment'].map({'negative': 0, 'positive': 1, 'neutral': 2}).tolist()
train_dataset = TweetDataset(train_tweets, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Tokenize and create DataLoader for testing set
test_tweets = test_df['text'].tolist()
test_labels = test_df['sentiment'].map({'negative': 0, 'positive': 1, 'neutral': 2}).tolist()
test_dataset = TweetDataset(test_tweets, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Fine-tuning parameters
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Fine-tune the model
print('Fine-tuning BERT on tweets... Starting Epochs\n')
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(batch['label'])

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(labels.cpu())
print('Finished Fine-tuning.\n')

# Evaluate the model on the testing set
model.eval()
test_predictions = []
test_true_labels = []

print('Evaluating BERT on tweets... Starting Predictions\n')
with torch.no_grad():
    for batch in test_dataloader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(batch['label'])

        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

        test_predictions.extend(predictions.tolist())
        test_true_labels.extend(labels.tolist())
        print(labels.cpu())
print('Finished Predictions.\n')

# Print evaluation metrics
accuracy = accuracy_score(test_true_labels, test_predictions)
classification_report_str = classification_report(test_true_labels, test_predictions)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_str)
