#2023/12/4 modified based on huggingfaceSequence5, remove NLP related
#ref： https://github.com/huggingface/transformers/blob/main/examples/pytorch/audio-classification/run_audio_classification.py
#https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb
from datasets import load_dataset, DatasetDict, features
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, DataCollatorWithPadding, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator, EvalPrediction)
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import TrainingArguments
import evaluate
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import math
import collections
import numpy as np
import random
import json
from random import randint
valkey='test'
import datetime

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def getlabels(raw_datasets, task_column, target_column):
    labels = raw_datasets["train"].features[target_column].names
    column_names = raw_datasets["train"].column_names
    print("column_names:", column_names)
    if task_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name not found in dataset. "
        )
    if target_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name not found in dataset. "
        )
    id2label_fn = raw_datasets["train"].features[target_column].int2str
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(labels))
    }
    label2id = {v: k for k, v in id2label.items()}
    #print("label2id: ", label2id)
    #Option2
    # label2id, id2label = dict(), dict()
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label
    # newtarget_column='label'
    # raw_datasets["train"][newtarget_column] = raw_datasets["train"].pop(target_column) #split_datasets.pop("test")
    # raw_datasets[valkey][newtarget_column] = raw_datasets[valkey].pop(target_column)
    # newcolumn_names = raw_datasets["train"].column_names
    columns_remove = []
    for column in column_names:
        if not (column==target_column or column==task_column):
            columns_remove.append(column)
    return labels, id2label, label2id, column_names, columns_remove

def loadmodel(model_checkpoint, id2label, label2id, task="audio-classification", pretrained="", cache_dir="", unfreezename="", return_attention_mask=True, freeze_feature_encoder=True, modelchange=False):
    ignore_mismatched_sizes = modelchange #when loading model, ignore size missmatch
    # label2id, id2label = {}, {}
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label
    
    # id2label_fn = gtzan["train"].features["genre"].int2str
    # id2label = {
    #     str(i): id2label_fn(i)
    #     for i in range(len(gtzan_encoded["train"].features["label"].names))
    # }
    # label2id = {v: k for k, v in id2label.items()}
    if cache_dir:
        mycache_dir = cache_dir
    elif os.environ.get('HF_HOME') is not None:
        mycache_dir = os.environ.get('HF_HOME')

    if modelchange: #task == "audio-classification":
        # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
        # transformer outputs in the classifier, but it doesn't always lead to better accuracy
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint, cache_dir=mycache_dir,return_attention_mask=return_attention_mask)
        # model_args.feature_extractor_name or model_args.model_name_or_path,
        # return_attention_mask=model_args.attention_mask,
        # cache_dir=model_args.cache_dir,
        #Option1
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=len(label2id), #len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task=task, #"audio-classification",
            cache_dir=mycache_dir,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            cache_dir=mycache_dir,
        )
    else:
        #Option2
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            cache_dir=mycache_dir,
        )
        #ignore_mismatched_sizes: Will enable to load a pretrained model whose head dimensions are different.

    starting_epoch = 0
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        starting_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['model_state_dict'])

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")

    # freeze the convolutional waveform encoder
    if freeze_feature_encoder:
        model.freeze_feature_encoder()
    modelparameters(model, unfreezename)
    return model, feature_extractor, starting_epoch

# max_length = 128
# def preprocess_function(examples):
#     inputs = [ex[source_lang] for ex in examples["translation"]] #1000
#     targets = [ex[target_lang] for ex in examples["translation"]] #1000
#     model_inputs = globaltokenizer(
#         inputs, text_target=targets, max_length=max_length, truncation=True
#     )
#     return model_inputs

def loaddata(args, USE_HPC):
    task_column =""
    text_column = ""
    target_column =""
    if args.data_type == "huggingface":
        if USE_HPC:
            if args.data_name=='kde4':
                #raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
                datasetpath=os.path.join(mycache_dir, args.data_name, "en-fr-lang1\=en\,lang2\=fr", "0.0.0", "/243129fb2398d5b0b4f7f6831ab27ad84774b7ce374cf10f60f6e1ff331648ac") #"/data/cmpe249-fa23/Huggingfacecache/imdb/plain_text/1.0.0/d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0"
                #raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir) #eli5
                datasetpath=os.path.join(mycache_dir, args.data_name)
                trainarrowpath=os.path.join(mycache_dir, args.data_name, args.data_name+'-train.arrow')
                #valarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-validation.arrow')
                #testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "en"
                target_column = "fr"
            elif args.data_name=='opus100':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'enzh')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                valarrowpath=os.path.join(datasetpath, args.data_name+'-validation.arrow')
                testarrowpath=os.path.join(datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                #raw_datasets = load_dataset("opus100", language_pair="en-zh")
                text_column =  "en"
                target_column = "zh"
            elif args.data_name == 'wmt19':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'zh-en-24b9c423f6ba2174/0.0.0/29e210fae5690e843cae5dc43b53db36c4e02f927db50cd5235a22ab42dde90a')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train*.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "en"
                target_column = "zh"
            elif args.data_name=='billsum': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, 'default/3.0.0/75cf1719d38d6553aa0e0714c393c74579b083ae6e164b2543684e3e92e0c4cc')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column = "text"
                target_column = "summary"
            elif args.data_name=='cnn_dailymail': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, '3.0.0/3.0.0/1b3c71476f6d152c31c1730e83ccb08bcf23e348233f4fcc11e182248e6bf7de')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train*.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "article" #("article", "highlights")
                target_column = "highlights"
            elif args.data_name=='xsum': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, 'default/1.2.0/082863bf4754ee058a5b6f6525d0cb2b18eadb62c7b370b095d1364050a52b71')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column = "document"
                target_column = "summary"
            elif args.data_name == "speech_commands": #AUDIO
                raw_datasets = DatasetDict()
                raw_datasets["train"] = load_dataset(args.data_name, args.dataconfig, split='train')
                #raw_datasets[valkey] = load_dataset(args.data_name, args.dataconfig, split='validation')
                #raw_datasets = load_dataset(args.data_name, args.dataconfig, split='test')
                task_column ="audio" #(['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id']
                text_column = "audio"
                target_column = "label"
            elif args.data_name == "marsyas/gtzan":
                raw_datasets = load_dataset(args.data_name, "all")
                task_column ="audio" 
                text_column = "audio"
                target_column = "genre"
            elif args.data_name =="common_language":
                raw_datasets = load_dataset(args.data_name)
                task_column ="audio" 
                text_column = "audio"
                target_column = "language" #['client_id', 'path', 'audio', 'sentence', 'age', 'gender', 'language']
            else:
                raw_datasets = load_dataset(args.data_name, language_pair=(args.target_lang,args.source_lang))
                text_column =  "en"
                target_column = "zh"
        else:
            if args.data_name=='kde4':
                raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
                task_column ="translation"
                text_column =  "en"
                target_column = "fr"
                #source_lang = args.source_lang.split("_")[0]
                #target_lang = args.target_lang.split("_")[0]
            elif args.data_name=='opus100':
                raw_datasets = load_dataset("opus100", language_pair="en-zh")
                task_column ="translation"
                text_column =  "en"
                target_column = "zh"
            elif args.data_name=='opus_books':
                raw_datasets = load_dataset("opus_books", "en-fr")
                task_column ="translation"
                text_column =  "en"
                target_column = "fr"
            elif args.data_name=='billsum': #summarization
                #raw_datasets = load_dataset("billsum", split="ca_test")
                raw_datasets = load_dataset("billsum")
                text_column = "text"
                target_column = "summary"
            elif args.data_name=='cnn_dailymail': #summarization
                raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
                text_column =  "article" #("article", "highlights")
                target_column = "highlights"
            elif args.data_name=='xsum': #summarization
                raw_datasets = load_dataset("xsum")
                text_column = "document"
                target_column = "summary"
            elif args.data_name in ['squad', 'squad_v2']: #QA
                raw_datasets = load_dataset(args.data_name)
                #raw_datasets = load_dataset("squad", split="train[:5000]") #'train', 'test'
                #raw_datasets["train"][0] #'id', 'title','context', 'question', 'answers' (dict with 'text' and 'answer_start'),  
                task_column ="question"
                text_column = "context"
                target_column = "answers"
            #AUDIO part
            elif args.data_name == "speech_commands": 
                raw_datasets = DatasetDict()
                raw_datasets["train"] = load_dataset(args.data_name, args.dataconfig, split='train')
                #raw_datasets[valkey] = load_dataset(args.data_name, args.dataconfig, split='validation')
                #raw_datasets = load_dataset(args.data_name, args.dataconfig, split='test')
                task_column ="audio" #(['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id']
                text_column = "audio"
                target_column = "label"
            elif args.data_name == "marsyas/gtzan":
                raw_datasets = load_dataset(args.data_name, "all")
                task_column ="audio" 
                text_column = "audio"
                target_column = "genre"
            elif args.data_name =="common_language":
                raw_datasets = load_dataset(args.data_name)
                task_column ="audio" 
                text_column = "audio"
                target_column = "language" #['client_id', 'path', 'audio', 'sentence', 'age', 'gender', 'language']
            elif args.data_name.endswith("minds14"):
                #https://huggingface.co/datasets/PolyAI/minds14 contains recordings of people asking an e-banking system questions in several languages and dialects, and has the intent_class for each recording
                if args.dataconfig:
                    subsetconfig = args.dataconfig
                else:
                    subsetconfig = "all" #"en-AU" "zh-CN"
                raw_datasets = load_dataset("PolyAI/minds14", name=subsetconfig, split="train")
                #raw_datasets = raw_datasets.train_test_split(test_size=0.2)
                #minds can be used to classify intent_class, lang_id, and speech recognition (english_transcription)
                #contains "path", "audio"dict("path", "array")
                task_column ="audio" 
                text_column = "path"
                if args.task=="audio-classification":
                    if args.subtask.startswith("intent"):
                        target_column = "intent_class"
                    else:
                        target_column = "lang_id"
                else:
                    target_column = "english_transcription"
            elif args.data_name == "superb":
                #https://huggingface.co/datasets/superb#ks
                if args.dataconfig:
                    subsetconfig = args.dataconfig
                else:
                    subsetconfig = "ks" #Keyword Spotting (KS)
                raw_datasets = load_dataset("superb", name=subsetconfig, split="train")
                task_column ="audio" 
                text_column = "file"
                target_column = "label"
            elif args.data_name == "google/fleurs":
                raw_datasets = load_dataset("google/fleurs", "all", split="train")
                task_column ="audio" 
                text_column = "path"
                target_column = "lang_id" #language
            else: 
                #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
                raw_datasets = load_dataset(args.data_name)
                text_column = "text"
                target_column = "summary"
        #Download to home/.cache/huggingface/dataset
        print(raw_datasets.column_names)
        #splits=raw_datasets.split
        # print(raw_datasets.columns)
        if isinstance(raw_datasets.column_names, dict):
            print("All keys in raw datasets:", raw_datasets['train']) #obly one ['translation'] key
            split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
        else: #no train/test split
            split_datasets = DatasetDict()
            split_datasets["train"] = raw_datasets
            split_datasets = split_datasets["train"].train_test_split(train_size=0.9, seed=20)
        
   
        # rename the "test" key to "validation" 
        #split_datasets["validation"] = split_datasets.pop("test")
        #one element
        if args.task=="translation":
            sampletext = split_datasets["train"][1][task_column]
            print("Translation text: ", sampletext)
        elif args.task=="summarization":
            sampletext_text = split_datasets["train"][1][text_column]
            sampletext_target = split_datasets["train"][1][target_column]
            print("Summarization context: ", sampletext_text)
            print("Summarization target: ", sampletext_target)
        elif args.task=="QA":
            oneexample = split_datasets["train"][1]
            print("Context: ", oneexample[text_column])
            print("Question: ", oneexample[task_column])
            print("Answer: ", oneexample[target_column])#dict with 'text' and 'answer_start'
        elif args.task.startswith("audio"):
            oneexample = split_datasets["train"][1]
            print("Audio: ", oneexample[task_column])
            print("Audio target: ", oneexample[target_column])
        raw_datasets = split_datasets
        if args.subset>0:
            if args.subset<1:
                trainlen=int(len(raw_datasets["train"])*args.subset)
                testlen=int(len(raw_datasets[valkey])*args.subset)
            else:
                trainlen = int(min(args.subset, len(raw_datasets["train"])))
                testlen = int(trainlen/10)
            print("trainlen:", trainlen)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(testlen))])
    
        #limit the evaluation set size
        maxtestlen = 5000
        if len(raw_datasets[valkey])>maxtestlen:
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(maxtestlen))])

    return raw_datasets, text_column, target_column, task_column

def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

from transformers import pipeline
def inferencesample(datasample, task, model, usepipeline=True, feature_extractor=None, device='cuda', columnname='audio'):
    sample = datasample[columnname] #raw_datasets["train"][0][task_column]
    print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    print(f"inputs keys: {list(inputs.keys())}")
    print(
        f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
    )
    if usepipeline:
        mypipeline = pipeline(
            task, #"audio-classification",
            model=model, #"anton-l/xtreme_s_xlsr_300m_minds14",
        )
        #sample = datasample[columnname]
        result = mypipeline(sample)#sample can be audio (sample["audio"]) or path (minds[0]["path"])
        print(result)
    else:
        #sample=datasample[columnname]
        #sample["audio"]["array"]
        print(feature_extractor.sampling_rate)
        print(sample['sampling_rate']) #dataset.features["audio"].sampling_rate   
        if isinstance(model, str):
            feature_extractor = AutoFeatureExtractor.from_pretrained(model)
            model = AutoModelForAudioClassification.from_pretrained(model)
            
        inputs = feature_extractor(sample["array"],sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        model=model.to(device)
        inputs=inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        #Get the class with the highest probability
        logitsmax=torch.argmax(logits)
        predicted_class_ids = logitsmax.item() #.item() moves the scalar data to CPU
        #use the model’s id2label mapping to convert it to a label:
        result = model.config.id2label[str(predicted_class_ids)]
        print(result)
    return result




class myEvaluator:
    def __init__(self, args, useHFevaluator=False, dualevaluator=False):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = args.task
        self.preds = []
        self.refs = []
        self.HFmetric = None
        if self.task.startswith("audio"):
            self.metricname = "accuracy"
            # Load the accuracy metric from the datasets package
            self.HFmetric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(self, eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.HFmetric.compute(predictions=predictions, references=eval_pred.label_ids)

    
    def compute(self, predictions=None, references=None):
        results = []
        if predictions is not None and references is not None:
            results = self.HFmetric.compute(predictions=predictions, references=references)
            #print("HF evaluator:", results)
        else: #evaluate the whole dataset
            results = self.HFmetric.compute()
            print("HF evaluator result1:", results)
            results2 = self.HFmetric.compute(predictions=self.preds, references=self.refs) #the same results
            print("HF evaluator result2:", results2)
            self.preds.clear()
            self.refs.clear()
        return results
    
    def add_batch(self, predictions, references):
        self.HFmetric.add_batch(predictions=predictions, references=references)
        #self.preds.append(predictions)
        self.refs.extend(references)
        self.preds.extend(predictions)
        #references: list of list
        # for ref in references:
        #     self.refs.append(ref[0])
        #print(len(self.refs))

def evaluate_dataset(model, eval_dataloader, device, metric):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        logits = outputs.logits #[4, 12]
        #Get the class with the highest probability
        logitsmax=torch.argmax(logits, dim=-1) #argmax (https://pytorch.org/docs/stable/generated/torch.argmax.html) for the last dimension, torch.Size([4])
        #predicted_class_ids = logitsmax.item() #.item() moves the data to CPU
        
        #use the model’s id2label mapping to convert it to a label:
        #decoded_preds = [model.config.id2label[str(pred.item())] for pred in logitsmax]
        decoded_preds = [pred.item() for pred in logitsmax]
        #result = model.config.id2label[str(predicted_class_ids)]

        labels = batch["labels"] #label is also integer
        decoded_labels = [label.item() for label in labels]
        
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # result1 = metric.compute(predictions=decoded_preds, references=decoded_labels)
        #evalmetric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    results = metric.compute()
    #evalresults = evalmetric.compute()
    print(f"Evaluation score: {results}")
    #print(evalresults['score'])
    return results

import shutil
def savemodels(model, optimizer, epoch, trainoutput):
    modelfilepath=os.path.join(trainoutput, 'savedmodel.pth')
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, modelfilepath)
    modelfilepathwithepoch=os.path.join(trainoutput, 'epoch'+str(epoch)+'_savedmodel.pth')
    shutil.copy(modelfilepath, modelfilepathwithepoch)
    #Huggingface format:
    model.save_pretrained(trainoutput)

#data_name="marsyas/gtzan", dataconfig="", model_checkpoint="ntu-spml/distilhubert"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #data related arguments
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="superb",
                    help='data name: superb, google/fleurs, minds14, common_language, marsyas/gtzan')
    parser.add_argument('--dataconfig', type=str, default='',
                    help='dataset_config_name, e.g., subset')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/DATA10T/Cache", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache"
    #model related arguments
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base",
                    help='Model checkpoint name from HF, anton-l/xtreme_s_xlsr_300m_minds14, facebook/wav2vec2-base, ntu-spml/distilhubert')
    parser.add_argument('--checkpointfolder', type=str, default="",
                    help='Model training checkpoint to resume')
    parser.add_argument('--modelchange', default=True, action='store_true', help='ignore model mismatch, allow model change') 
    parser.add_argument('--task', type=str, default="audio-classification",
                    help='tasks: audio-classification, openqa, translation, summarization, QA')
    parser.add_argument('--subtask', type=str, default="intent-classification",
                    help='Sub tasks')
    parser.add_argument('--hfevaluate', default=True, action='store_true',
                    help='perform evaluation via HFevaluate or localevaluate')
    parser.add_argument('--dualevaluate', default=False, action='store_true',
                    help='perform evaluation via HFevaluate and localevaluate')
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    #training related arguments
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="0109",
                    help='Name the current training')
    parser.add_argument('--training', default=True, action='store_true',
                    help='Perform training')
    parser.add_argument('--usehpc', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', default=False, action='store_true',
                    help='Use Huggingface accelerator')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=16, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=3e-5, type=float, help='Learning rate')
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--max_length_seconds",
        type=int,
        default=5, #20,
        help=(
            "Audio clips will be randomly cut to this length during training if the value is set.."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help=(
            "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384, #128, #1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    args = parser.parse_args()

    print("useHFevaluator:", args.hfevaluate)
    print("dualevaluator:", args.dualevaluate)

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    use_accelerator = args.useHFaccelerator
    model_checkpoint = args.model_checkpoint
    use_fp16 = True

    USE_HPC=args.usehpc
    if USE_HPC:
        #https://huggingface.co/docs/transformers/installation#offline-mode
        #HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
        #os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        #os.environ['HF_EVALUATE_OFFLINE'] = "1"
        #os.environ['HF_DATASETS_OFFLINE'] = "1"
        #os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        os.environ['https_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTPS_PROXY'] = "http://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        #taskname=args.traintag #"eli5asksciencemodeling"
    else:
        if os.path.exists(args.data_path):
            mycache_dir=args.data_path
            os.environ['HF_HOME'] = mycache_dir
        elif os.environ.get('HF_HOME') is not None:
            mycache_dir=os.environ['HF_HOME']
        else:
            mycache_dir="./data/"
        # mycache_dir=os.path.join('D:',os.sep, 'Cache','huggingface')
        
        print("HF_HOME:", os.environ['HF_HOME'])
        # os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # if os.environ.get('HF_HOME') is None:
        #     mycache_dir=args.data_path
        #     os.environ['HF_HOME'] = mycache_dir
        #     os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # else:
        #     print("HF_HOME:", os.environ['HF_HOME'])
        #     mycache_dir=os.environ['HF_HOME']
        trainoutput=args.outputdir #"./output"
        #taskname=args.traintag #taskname="eli5asksciencemodeling"
    
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    raw_datasets, text_column, target_column, task_column = loaddata(args, USE_HPC)
    labels, id2label, label2id, column_names, columns_remove = getlabels(raw_datasets, task_column, target_column)
    
    
    model, feature_extractor, starting_epoch = loadmodel(model_checkpoint, id2label, label2id, task=task, pretrained=args.pretrained, unfreezename=args.unfreezename, return_attention_mask=True, freeze_feature_encoder=True, modelchange=args.modelchange)
    model_input_name = feature_extractor.model_input_names[0]
    print("model_input_name:", model_input_name) #input_values
    model = model.to(device)
    
    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        task_column, features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    datasample=raw_datasets["train"][0]
    result=inferencesample(datasample=datasample, task = args.task, model=model, usepipeline=False, feature_extractor=feature_extractor, device=device, columnname=task_column)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples[task_column]] #"audio"
        target_labels = [np.int64(x) for x in examples[target_column]]
        #https://huggingface.co/docs/transformers/main_classes/feature_extractor
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * args.max_length_seconds),
            truncation=True,
            padding=True, #'max_length'
            return_attention_mask=True,
        )
        #return inputs
        output_batch = inputs #{model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = target_labels #list(target_labels) #list(examples[target_column])
        return output_batch

    def preprocess_function_simple(examples):
        audio_arrays = [x["array"] for x in examples[task_column]] #examples[task_column] is list of audio data
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
            truncation=True,
            padding=True #'max_length'
        )
        return inputs
    
    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        subsampled_wavs = []
        #print(batch.keys())
        #print(batch[task_column])
        if isinstance(batch[task_column], list):
            for audio in batch[task_column]:
                wav = random_subsample(
                    audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
                subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs, 
                                       sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
                                       truncation=True,
                                       padding=True)
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[target_column])
        else:
            audio = batch[task_column]
            wav = random_subsample(
                    audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs,sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
                                       truncation=True,
            padding=True) #'max_length')
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = batch[target_column]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        if isinstance(batch[task_column], list):
            wavs = [audio["array"] for audio in batch[task_column]]
        else:
            audio = batch[task_column]
            wavs = [audio["array"]]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), truncation=True, padding=True)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[target_column])

        return output_batch

    samplebatch1 = preprocess_function_simple(raw_datasets['train'][:5])
    samplebatch2 = train_transforms(raw_datasets['train'][:5])
    processmode=1
    if processmode == 1:
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms, output_all_columns=True)
        #transferred_datasets = DatasetDict()
        #transferred_datasets["train"]=raw_datasets["train"].map(train_transforms)
        # # Set the validation transforms
        raw_datasets[valkey].set_transform(val_transforms, output_all_columns=True)
        # #transferred_datasets[valkey]=raw_datasets[valkey].map(val_transforms)
        # train_dataset=raw_datasets["train"]
        # eval_dataset=raw_datasets[valkey]
        # #train_dataset=transferred_datasets["train"]
        # #eval_dataset=transferred_datasets[valkey]
    elif processmode == 2:
        raw_datasets = raw_datasets.map(
            preprocess_function,
            remove_columns=raw_datasets["train"].column_names, #["audio", "file"],
            batched=True,
            batch_size=100,
            num_proc=1,
        )#Get "labels" and inputs
        #dataset_encoded = dataset_encoded.rename_column("genre", "label")
    else:
        columns_remove.append("audio")
        print("columns_remove:", columns_remove)
        #https://huggingface.co/docs/datasets/about_map_batch
        raw_datasets = raw_datasets.map(
            preprocess_function_simple,
            remove_columns=columns_remove,
            batched=True, #The primary objective of batch mapping is to speed up processing. The default batch size is 1000
            batch_size=100,
        )
        column_names = raw_datasets["train"].column_names
        target_ready = False
        input_ready = False
        for column in column_names:
            if column == target_column:
                if target_column != "label":
                    #change target_column name
                    dataset_encoded = dataset_encoded.rename_column(target_column, "label")
                    target_column="label"
                    target_ready = True
                else:
                    target_ready = True
            elif column == model_input_name:
                input_ready = True
            else:#remove column
                print("column name:", column)
        column_names = raw_datasets["train"].column_names
        print(column_names)


    # Load the accuracy metric from the datasets package
    #metric = evaluate.load("accuracy")
    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    # def compute_metrics(eval_pred):
    #     """Computes accuracy on a batch of predictions"""
    #     predictions = np.argmax(eval_pred.predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    metriceval = myEvaluator(args, useHFevaluator=args.hfevaluate, dualevaluator=args.dualevaluate)
    #metriceval.compute_metrics
    result1=metriceval.compute(predictions=[2,7,6,1], references=[2,7,7,7])
    result2=metriceval.compute(predictions=[2,7,6,1,1], references=[2,7,6,1,1])

    usehftrainer = True
    if usehftrainer:
        training_args = TrainingArguments(
            trainoutput,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.total_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            #fp16=True,
            push_to_hub=False,
        )

        # Initialize our trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets[valkey],
            compute_metrics=metriceval.compute_metrics,
            tokenizer=feature_extractor,
        )

        
        if args.checkpointfolder and os.path.exists(args.checkpointfolder):
            checkpoint = args.checkpointfolder #"output/facebook/wav2vec2-base/minds14_0105/checkpoint-14704"
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        #Perform evaluation
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    else:
        train_dataloader = DataLoader(
            raw_datasets["train"],
            shuffle=True,
            collate_fn=None,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            raw_datasets[valkey], collate_fn=None, batch_size=args.batch_size
        )

        results=evaluate_dataset(model, eval_dataloader, device, metriceval)

        # Optimizer
        adam_beta1=0.9
        adam_beta2=0.999
        adam_epsilon=1e-8
        optimizer = AdamW(
            list(model.parameters()),
            lr=args.learningrate,
            betas=[adam_beta1, adam_beta2],
            eps=adam_epsilon,
        )
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.total_epochs * num_update_steps_per_epoch

        num_warmup_steps = 10
        lr_scheduler = get_scheduler(
                name="linear", #["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_train_steps,
            )
        
        #recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps /  num_update_steps_per_epoch)
        total_batch_size = args.batch_size * args.gradient_accumulation_steps

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0
        starting_epoch = 0
        for epoch in range(starting_epoch, num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                #backward
                loss.backward()

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1
            #Evaluation
            results=evaluate_dataset(model, eval_dataloader, device, metriceval)   
            print(f"epoch {epoch}, loss: {loss}, evaluation: {metriceval.metricname}")
            print("Evaluation result:", results)
            # Save the results
            with open(os.path.join(trainoutput, f"epoch{epoch}_"+"eval_results.json"), "w") as f:
                #json.dump({"eval_bleu": results["score"]}, f)
                json.dump(results, f)
            
            savemodels(model, optimizer, epoch, trainoutput)
            #feature_extractor.save_pretrained(trainoutput)
    
    # using now() to get current time
    current_time = datetime.datetime.now()
    # Printing value of now.
    print("Time now is:", current_time)
    print("Finished")


    

