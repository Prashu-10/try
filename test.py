import pandas as pd

test_df = pd.read_csv('data/test.csv')

print("length of testing set: "+str(len(test_df)))

test_text = list(test_df.text)
import re

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'@\S+','',s)
    return re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", s)

for i in range(len(test_text)):
    test_text[i] = clean_text(test_text[i])
    test_text[i] = test_text[i].lower()

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("results/checkpoint-3500")
model = DistilBertForSequenceClassification.from_pretrained("results/checkpoint-3500")

preds=[]

for i in test_text:
    inputs = tokenizer(i , return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(predicted_class_id)
    preds.append(predicted_class_id)

submission = pd.DataFrame()
submission['id']=test_df['id']
submission['target']=preds
submission.to_csv('submission.csv',index=False)

