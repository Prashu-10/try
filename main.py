import numpy as np
import os
import pandas as pd

for dirname, _, filenames in os.walk('/Users/pranjalbharti/PycharmProjects/NLP_tweet_disaster_classification/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
sample_df = pd.read_csv('data/sample_submission.csv')

print("length of training set: "+str(len(train_df)))
print("length of testing set: "+str(len(test_df)))
print("length of sample submission set: "+str(len(sample_df)))


from sklearn.model_selection import train_test_split
train,val= train_test_split(train_df , test_size=0.20, random_state=42)
print(len(train))
print(len(val))
# data cleaning
import re

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", s)

train_text = list(train.text)
train_label = list(train.target)
val_text = list(val.text)
val_label = list(val.target)
test_text = list(test_df.text)

for i in range(len(train_text)):
    train_text[i] = clean_text(train_text[i])
    train_text[i] = train_text[i].lower()

for i in range(len(val_text)):
    val_text[i] = clean_text(val_text[i])
    val_text[i] = val_text[i].lower()


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased/')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased/', num_labels=2)

train_encodings = tokenizer(train_text,truncation=True,padding=True)
val_encodings = tokenizer(val_text,truncation=True,padding=True)
test_encodings = tokenizer(test_text,truncation=True,padding=True)


import torch
class Disaster_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Disaster_Dataset(train_encodings, train_label)
val_dataset = Disaster_Dataset(val_encodings, val_label)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

