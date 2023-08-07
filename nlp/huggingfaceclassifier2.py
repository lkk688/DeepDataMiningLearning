#sequence classifier: https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt
#token_classifier: https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt
#The traditional framework used to evaluate token classification prediction is seqeval. pip install seqeval
#ref: https://huggingface.co/transformers/v4.1.1/custom_datasets.html

from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_scheduler
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import evaluate
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def tokenclassifier_evaluation(metric, raw_datasets):
    #metric = evaluate.load("seqeval") #"seqeval"

    labels = raw_datasets["train"][0]["ner_tags"]
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    labels = [label_names[i] for i in labels]
    print(labels)

    predictions = labels.copy()
    predictions[2] = "O"
    #create fake predictions for those by just changing the value at index 2:
    testmetricresult = metric.compute(predictions=[predictions], references=[labels])
    print(testmetricresult)

    return testmetricresult

def tokenclassifier_metrics(eval_preds, metric, label_names):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def evaluation(dataset_name, model, eval_dataloader, device):
    if task == "token_classifier":
        metric = evaluate.load("seqeval") #"seqeval"
    elif task == "sequence_classifier":
        metric = evaluate.load("glue", "mrpc")
    
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metricresult = metric.compute()
    return metricresult

def checkdataset(raw_datasets):
    if "train" in raw_datasets.keys():
        print("Train len:", len(raw_datasets["train"]))
        oneitem = raw_datasets["train"][0]
        print("All keys in oneitem:", oneitem.keys())
        print(oneitem["tokens"])
        print(oneitem["ner_tags"])
        ner_feature = raw_datasets["train"].features["ner_tags"]
        print(ner_feature)
        #B means beginning, I means inside
        #PER: person, ORG: organization, LOC: location, MISC: miscellaneous
        label_names = ner_feature.feature.names
        print(label_names)

        words = oneitem["tokens"]
        labels = oneitem["ner_tags"]
        line1 = ""
        line2 = ""
        for word, label in zip(words, labels):
            full_label = label_names[label]
            max_length = max(len(word), len(full_label))
            line1 += word + " " * (max_length - len(word) + 1)
            line2 += full_label + " " * (max_length - len(full_label) + 1)

        print(line1)
        print(line2)

def checktokenizer(raw_datasets, tokenizer):
    inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
    print(inputs.tokens())
    #the tokenizer added the special tokens used by the model ([CLS] at the beginning and [SEP] at the end), some words are tokenized into two subwords
    print(inputs.word_ids())

    labels = raw_datasets["train"][0]["ner_tags"]
    word_ids = inputs.word_ids()
    print(labels)
    print(align_labels_with_tokens(labels, word_ids))
    #our function added the -100 for the two special tokens at the beginning and the end, and a new 0 for our word that was split into two tokens.

def imdbchecktokenizer(raw_datasets, tokenizer):
    train_texts=raw_datasets["train"]['text'] #25000 array
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    print(train_encodings)

#by default -100 is an index that is ignored in the loss function
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    if task == "sequence_classifier":
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    elif task == "token_classifier":
        return tokenize_and_align_labels(example)
    elif task =="sentiment":
        return tokenizer(example["text"], truncation=True, padding=True)
    else:
        return tokenizer(example["text"], truncation=True, padding=True, max_length=512)

def testdatacollator(data_collator, tokenized_datasets):
    print(tokenized_datasets["train"][0])
    batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
    #print(batch["labels"])
    testbatch={k: v.shape for k, v in batch.items()}
    print(testbatch)
    #compare this to the labels for the first and second elements in our dataset
    # for i in range(2):
    #     print(tokenized_datasets["train"][i]["labels"])
        #the second set of labels has been padded to the length of the first one using -100s.


#mrpc: MRPC (Microsoft Research Paraphrase Corpus) dataset
#The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_customdata_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            with open(text_file, mode="r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                #texts.append(text_file.read_text())
                labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels): 
        super(CustomModel,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(checkpoint, config=config)
        self.dropout = nn.Dropout(0.1) 
        self.vecsize = 768
        self.classifier = nn.Linear(self.vecsize,num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:,0,:].view(-1,self.vecsize)) # calculate losses
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="conll2003",
                    help='data name: imdb, conll2003, "glue", "mrpc" ')
    parser.add_argument('--dataconfig', type=str, default=None,
                    help='data config name in sequence classification, glue: "mrpc", None ')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default=r"E:\Dataset\NLPdataset\aclImdb",
                    help='path to get data')
    parser.add_argument('--model_checkpoint', type=str, default="bert-base-cased",
                    help='Model checkpoint name from https://huggingface.co/models, "bert-base-cased", "distilbert-base-uncased" "cardiffnlp/twitter-roberta-base-emotion"')
    parser.add_argument('--task', type=str, default="token_classifier",
                    help='NLP tasks: sentiment, token_classifier, "sequence_classifier", custom_classifier')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="0807",
                    help='Name the current training')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    model_checkpoint = args.model_checkpoint
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_len=512

    if args.data_type == "huggingface":
        if not args.dataconfig:
            raw_datasets = load_dataset(args.data_name)
        else:
            raw_datasets = load_dataset(args.data_name, args.dataconfig) #("glue", "mrpc") 
        #Download to home/.cache/huggingface/dataset

        print("All keys in raw datasets:", raw_datasets.keys())
        if "train" in raw_datasets.keys():
            print("Train len:", len(raw_datasets["train"]))
            oneitem = raw_datasets["train"][0]
            print("oneitem all keys:", oneitem.keys())
            features = raw_datasets["train"].features
            print("all features:", features) 
            if "label" in features.keys():
                classlabel=features['label'] #datasets.ClassLabel
                num_classes = classlabel.num_classes
                print(classlabel.names)
                print(classlabel.int2str(0)) #str2int
            elif "ner_tags" in features.keys(): #token classifier
                ner_feature = features["ner_tags"]
                num_classes = ner_feature.feature.num_classes
                print(ner_feature.feature.names) #['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        if "validation" in raw_datasets.keys():
            valkeyname="validation"
        elif "test" in raw_datasets.keys():
            valkeyname="test"
        
        if args.subset>0:
            trainlen=int(len(raw_datasets["train"])*args.subset)
            vallen=int(len(raw_datasets[valkeyname])*args.subset)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets[valkeyname] = raw_datasets[valkeyname].shuffle(seed=42).select([i for i in list(range(vallen))])
        
        #To preprocess our whole dataset, we need to tokenize all the inputs and apply align_labels_with_tokens() on all the labels
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets.set_format("torch")
        #tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
        if task == "sequence_classifier":
            tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #only pads the inputs
        elif task == "token_classifier":
            checkdataset(raw_datasets)
            checktokenizer(raw_datasets, tokenizer)
            tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) #labels should be padded the exact same way
        elif task == "custom_classifier":
            tokenized_datasets = tokenized_datasets.remove_columns(['text', 'token_type_ids'])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        else:
            #imdbchecktokenizer(raw_datasets, tokenizer)
            tokenized_datasets = tokenized_datasets.remove_columns(['text']) #['input_ids', 'attention_mask']
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #only pads the inputs
        print(tokenized_datasets["train"].column_names) #['labels', 'input_ids', 'attention_mask']
        testdatacollator(data_collator, tokenized_datasets)

    elif args.data_type == "custom":
        print("Custom dataset")
        train_texts, train_labels = read_customdata_split(os.path.join(args.data_path, 'train'))
        test_texts, test_labels = read_customdata_split(os.path.join(args.data_path, 'test'))

        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        tokenized_datasets = {}
        tokenized_datasets["train"] = CustomDataset(train_encodings, train_labels)
        valkeyname="validation"
        tokenized_datasets[valkeyname] = CustomDataset(val_encodings, val_labels)
        tokenized_datasets["test"] = CustomDataset(test_encodings, test_labels)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #only pads the inputs

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets[valkeyname], batch_size=args.batch_size, collate_fn=data_collator
    )

    for batch in train_dataloader:
        break
    #print(batch['input_ids'])
    testbatch={k: v.shape for k, v in batch.items()}
    print(testbatch)

    if task == "sequence_classifier" or task == "sentiment":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)
        #model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        print(model.config.num_labels) #9
    elif task == "token_classifier":
        ner_feature = raw_datasets["train"].features["ner_tags"]
        label_names = ner_feature.feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
        print(model.config.num_labels) #9
    elif task == "custom_classifier":
        model = CustomModel(model_checkpoint, num_labels=num_classes)
    

    #test forward
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape) #[8, 50, 9]

    optimizer = AdamW(model.parameters(), lr=args.learningrate) #5e-5

    #use a classic linear schedule from the learning rate to 0
    num_epochs = args.total_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("Using device:", device)

    if args.training == True:
        progress_bar = tqdm(range(num_training_steps))
        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                #sequence classification: outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        #Save models
        outputpath=os.path.join(args.outputdir, task, args.data_name+'_'+args.traintag)
        tokenizer.save_pretrained(outputpath)
        torch.save(model.state_dict(), os.path.join(outputpath, 'savedmodel.pth'))
        #model.load_state_dict(torch.load(PATH))
    else:
        #load saved model
        outputpath=os.path.join(args.outputdir, task, args.data_name+'_'+args.traintag)
        model.load_state_dict(torch.load(os.path.join(outputpath, 'savedmodel.pth')))

    #https://huggingface.co/docs/evaluate/index
    if task == "token_classifier":
        metric = evaluate.load("seqeval") #"seqeval"
        tokenclassifier_evaluation(metric, raw_datasets)
    elif task == "sequence_classifier" or task == "sentiment":
        metric = evaluate.load("glue", "mrpc")
    #elif task == "sentiment":
    else:
        #metric = load_metric("accuracy") #load_metric is deprecated
        #metric_f1 = load_metric("f1")
        metric = evaluate.load("accuracy") #"precision"
        print("metric test:", metric.compute(references=[0, 1], predictions=[0, 1]))
    
    model.eval()
    num_val_steps = len(eval_dataloader)
    valprogress_bar = tqdm(range(num_val_steps))
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.detach().cpu().clone().numpy()
        labels = batch["labels"]
        labels = labels.detach().cpu().clone().numpy()

        if task == "token_classifier":
            # Remove ignored index (special tokens) and convert to labels
            labels = [[label_names[l] for l in label if l != -100] for label in labels]
            predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

        metric.add_batch(predictions=predictions, references=labels)
        valprogress_bar.update(1)

    results = metric.compute()
    print(
        f" task {task}:",
        {
            key: results[f"{key}"]
            for key in results.keys()
        },
    )
    # print(
    #     f"epoch {epoch}:",
    #     {
    #         key: results[f"overall_{key}"]
    #         for key in ["precision", "recall", "f1", "accuracy"]
    #     },
    # )
    # metricresult = evaluation(model, eval_dataloader, device)
    # print(metricresult)

    
