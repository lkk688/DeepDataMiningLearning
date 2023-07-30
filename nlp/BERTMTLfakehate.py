#ref: https://www.analyticsvidhya.com/blog/2023/06/building-a-multi-task-model-for-fake-and-hate-probability-prediction-with-bert/

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from transformers import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import BertModel
import os

MAX_LEN = 256 # Define the maximum length of tokenized texts

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        
        self.fake_classifier = nn.Linear(768, 2)
        self.hate_classifier = nn.Linear(768, 2)
        self.sentiment_classifier = nn.Linear(768, 2)
        
        self.fake_softmax = nn.Softmax(dim=1)
        self.hate_softmax = nn.Softmax(dim=1)
        self.sentiment_softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids, attention_mask=attention_mask)
      pooled_output = outputs[1]
      pooled_output = self.dropout(pooled_output)

      fake_logits = self.fake_classifier(pooled_output)
      hate_logits = self.hate_classifier(pooled_output)
      sentiment_logits = self.sentiment_classifier(pooled_output)

      fake_probs = self.fake_softmax(fake_logits)
      hate_probs = self.hate_softmax(hate_logits)
      sentiment_probs = self.sentiment_softmax(sentiment_logits)

      return fake_logits, hate_logits, sentiment_logits, fake_probs , hate_probs, sentiment_probs

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding=True)


    # train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), 
    #                         torch.tensor(train_fake_labels), torch.tensor(train_hate_labels),
    #                         torch.tensor(train_sentiment_labels))
class MTLCustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels1, labels2, labels3):
        self.encodings = encodings
        #self.train_masks = train_masks
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item['train_masks'] = torch.tensor(self.train_masks[idx])
        item['labels1'] = torch.tensor(self.labels1[idx])
        item['labels2'] = torch.tensor(self.labels2[idx])
        item['labels3'] = torch.tensor(self.labels3[idx])
        return item

    def __len__(self):
        return len(self.labels1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="squad",
                    help='data name: imdb, conll2003, "glue", "mrpc" ')
    parser.add_argument('--data_path', type=str, default='./sampledata/fake-hate.csv',
                    help='path to get data')
    parser.add_argument('--model_checkpoint', type=str, default="bert-base-uncased",
                    help='Model checkpoint name from https://huggingface.co/models, "bert-base-cased"')
    parser.add_argument('--task', type=str, default="QA",
                    help='NLP tasks: sentiment, token_classifier, "sequence_classifier"')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--training', type=bool, default=False,
                    help='Perform training')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task

    df = pd.read_csv(args.data_path) 
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataset
    print(df.head())

    #Rename columns of the data frame. The column ‘label_f’ is renamed to ‘fake’, column ‘label_h’ is renamed to ‘hate’, and column ‘label_s’ is renamed to ‘sentiment’.
    df=df.rename(columns={'label_f':'fake','label_h':'hate','label_s':'sentiment'})
    print(df.head())

    # Define Task-specific Labels
    fake_labels = np.array(df['fake']) #(10466,)
    hate_labels = np.array(df['hate'])
    sentiment_labels = np.array(df['sentiment'])

    train_texts=df['text'].values.tolist() #np.array(df['text'])
    

    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['text']]
    
    #View a random text and tokenize it.
    print(df['text'][20])
    # rajneeti ko gandhwa diya ha in sapa congress ne I hate this type of rajneeti
    print(tokenizer.tokenize(df['text'][20]))


    #create a binary tensor with the same shape as the input sequence, serving as an attention mask
    #The tokens with a value of 1 represent actual tokens, while tokens with a value of 0 represent padding tokens. 
    # Using attention masks, the model will only focus on relevant information and helps improve the models’ efficiency and effectiveness.
    # Split the data into train and test sets
    train_inputs, test_inputs, train_fake_labels, test_fake_labels, \
    train_hate_labels, test_hate_labels, train_sentiment_labels, \
    test_sentiment_labels = train_test_split(train_texts, fake_labels, hate_labels, 
                            sentiment_labels, random_state=42, test_size=0.2) #10466->8732,2094
    
    #train_encodings = tokenizer(train_inputs, truncation=True, padding=True)
    #test_encodings = tokenizer(test_inputs, truncation=True, padding=True)

    
    # Pad and truncate the input_ids and attention_mask to a fixed length
    max_length = 256
    # train_inputs = pad_sequences(train_inputs, maxlen=max_length, dtype='long', 
    #                             value=0, truncating='post', padding='post')
    # test_inputs = pad_sequences(test_inputs, maxlen=max_length, dtype='long', 
    #                             value=0, truncating='post', padding='post')
    # train_masks = pad_sequences(train_masks, maxlen=max_length, dtype='long', 
    #                             value=0, truncating='post', padding='post')
    # test_masks = pad_sequences(test_masks, maxlen=max_length, dtype='long', 
    #                             value=0, truncating='post', padding='post')

    train_encodings = tokenizer(train_inputs, truncation=True, padding=True)
    test_encodings = tokenizer(test_inputs, truncation=True, padding=True)
    #'input_ids' 'attention_mask' 'token_type_ids'

    # Create attention masks
    #train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in train_encodings]
    #test_masks = [[int(token_id > 0) for token_id in input_id] for input_id in test_encodings]


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #only pads the inputs
    #Define Dataloader
    batch_size = 32

    train_data = MTLCustomDataset(train_encodings, train_fake_labels, train_hate_labels, train_sentiment_labels)
    # train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), 
    #                         torch.tensor(train_fake_labels), torch.tensor(train_hate_labels),
    #                         torch.tensor(train_sentiment_labels))
    train_sampler = RandomSampler(train_data)
    #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )

    test_data = MTLCustomDataset(test_encodings, test_fake_labels, test_hate_labels, test_sentiment_labels)
    # test_data = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_masks), 
    #                         torch.tensor(test_fake_labels), torch.tensor(test_hate_labels),
    #                         torch.tensor(test_sentiment_labels))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size, collate_fn=data_collator)

    for batch in train_dataloader:
        break
    testbatch={k: v.shape for k, v in batch.items()}
    print(testbatch)

    # Define Loss Function and Optimizer
    model = MultiTaskModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=2e-5)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    if args.training == True:
        num_epochs = args.total_epochs
        for epoch in range(num_epochs):
            for batch in train_dataloader: #'list' object has no attribute 'keys'
                model.train()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fake_labels = batch['labels1'].to(device)
                hate_labels = batch['labels2'].to(device)
                sentiment_labels = batch['labels3'].to(device)

                optimizer.zero_grad()

                fake_logits, hate_logits, sentiment_logits, fake_probs, \
                    hate_probs,sentiment_probs = model(input_ids, attention_mask)

                fake_loss = criterion(fake_logits, fake_labels)
                hate_loss = criterion(hate_logits, hate_labels)
                sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                loss = fake_loss + hate_loss + sentiment_loss

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        #outputpath=os.path.join(args.outputdir, task, args.data_name)
        torch.save(model.state_dict(), os.path.join(args.outputdir, 'savedmodel.pth'))
        torch.save({'tokenizer': tokenizer}, os.path.join(args.outputdir, 'savedmodel_info.pth'))
    else:
        #load saved model
        model.load_state_dict(torch.load(os.path.join(args.outputdir, 'savedmodel.pth')))

    model.eval()
    predictions = []
    num_val_steps = len(test_dataloader)
    valprogress_bar = tqdm(range(num_val_steps))
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fake_labels = batch['labels1'].to(device)
            hate_labels = batch['labels2'].to(device)
            sentiment_labels = batch['labels3'].to(device)

            #batch = tuple(t.to(device) for t in batch)
            #input_ids, attention_mask, fake_labels, hate_labels, sentiment_labels = batch
            
            fake_logits, hate_logits, sentiment_logits, fake_probs1 , hate_probs1, sentiment_probs1= \
                model(input_ids, attention_mask)
        
            fake_probs = nn.Softmax(dim=1)(fake_logits)
            hate_probs = nn.Softmax(dim=1)(hate_logits)
            sentiment_probs = nn.Softmax(dim=1)(sentiment_logits)
        
            for i in range(len(fake_probs)):
                predictions.append({
                    'text': tokenizer.decode(input_ids[i]),
                    'fake': fake_probs[i].tolist(),
                    'hate': hate_probs[i].tolist(),
                    'sentiment': sentiment_probs[i].tolist()
                })
            valprogress_bar.update(1)
    
    for i in range(len(predictions)):
        print('Text: {}'.format(predictions[i]['text']))
        print('Fake Probabilities: {}'.format(predictions[i]['fake']))
        print('Hate Probabilities: {}'.format(predictions[i]['hate']))
        print('Sentiment Probabilities: {}'.format(predictions[i]['sentiment']))
        print('-----------------------')