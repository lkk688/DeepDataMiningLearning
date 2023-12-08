#2023/12/4 modified based on huggingfaceSequence5, remove NLP related

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


def loadmodel(model_checkpoint, id2label, label2id, task="audio-classification", mycache_dir="", pretrained="", hpc=True, unfreezename="", return_attention_mask=True, freeze_feature_encoder=True, ignore_mismatched_sizes=False):
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
    
    if hpc==True:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        if task == "audio-classification":
            feature_extractor = AutoFeatureExtractor.from_pretrained(localpath, do_normalize=True, 
                                                                    return_attention_mask=return_attention_mask)
            config = AutoConfig.from_pretrained(
                localpath,
                num_labels=len(labels),
                label2id=label2id,
                id2label=id2label,
                finetuning_task=task, #"audio-classification",
                cache_dir=mycache_dir
            )
            model = AutoModelForAudioClassification.from_pretrained(
                localpath,
                config=config,
                cache_dir=mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
            #ignore_mismatched_sizes: Will enable to load a pretrained model whose head dimensions are different.
    else:
        if task == "audio-classification":
            # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
            # transformer outputs in the classifier, but it doesn't always lead to better accuracy
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint, 
                                                                    return_attention_mask=return_attention_mask,
                                                                    cache_dir=mycache_dir)
            # model_args.feature_extractor_name or model_args.model_name_or_path,
            # return_attention_mask=model_args.attention_mask,
            # cache_dir=model_args.cache_dir,
            config = AutoConfig.from_pretrained(
                model_checkpoint,
                num_labels=len(labels),
                label2id=label2id,
                id2label=id2label,
                finetuning_task=task, #"audio-classification",
                cache_dir=mycache_dir
            )
            model = AutoModelForAudioClassification.from_pretrained(
                model_checkpoint,
                config=config,
                cache_dir=mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
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
                #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
                raw_datasets = load_dataset(args.data_name)
                text_column = "text"
                target_column = "summary"
        #Download to home/.cache/huggingface/dataset
        
        print("All keys in raw datasets:", raw_datasets['train'][0].keys()) #obly one ['translation'] key
        split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
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
            print("Audio: ", oneexample[text_column])
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




class myEvaluator:
    def __init__(self, args, useHFevaluator=False, dualevaluator=False):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = args.task
        self.preds = []
        self.refs = []

    
    def compute(self, predictions=None, references=None):
        results = []
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator==True or self.dualevaluator==True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        
        if self.useHFevaluator==False or self.dualevaluator==True:
            #self.preds.append(predictions)
            self.refs.extend(references)
            self.preds.extend(predictions)
            #references: list of list
            # for ref in references:
            #     self.refs.append(ref[0])
            #print(len(self.refs))

def evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, num_beams, metric):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    #evalprogress_bar = tqdm(range(num_training_steps))
    model.eval()
    gen_kwargs = {
        "max_length": max_target_length,
        "num_beams": num_beams,
    }
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            if not use_accelerator:
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            #evalprogress_bar.update(1)
        labels = batch["labels"]
        if use_accelerator:
            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)

            #decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered, ignore_pad_token_for_loss)
            decoded_preds=[]
            decoded_labels=[]
        else:
            #decoded_preds, decoded_labels = postprocess(generated_tokens, labels, ignore_pad_token_for_loss)
            decoded_preds=[]
            decoded_labels=[]
        
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        #evalmetric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    results = metric.compute()
    #evalresults = evalmetric.compute()
    #print(f"BLEU score: {results['score']:.2f}")
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
    parser.add_argument('--data_name', type=str, default="common_language",
                    help='data name: marsyas/gtzan')
    parser.add_argument('--dataconfig', type=str, default='v0.02',
                    help='dataset_config_name')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache",
                    help='path to get data ') #r"E:\Dataset\NLPdataset\aclImdb"
    #model related arguments
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base",
                    help='Model checkpoint name from HF, "facebook/wav2vec2-base", ntu-spml/distilhubert') 
    parser.add_argument('--task', type=str, default="audio-classification",
                    help='NLP tasks: openqa, translation, summarization, QA')
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
    parser.add_argument('--traintag', type=str, default="1204",
                    help='Name the current training')
    parser.add_argument('--training', default=True, action='store_true',
                    help='Perform training')
    parser.add_argument('--usehpc', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', default=False, action='store_true',
                    help='Use Huggingface accelerator')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
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
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--max_length_seconds",
        type=int,
        default=20,
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
        os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        os.environ['HF_EVALUATE_OFFLINE'] = "1"
        os.environ['HF_DATASETS_OFFLINE'] = "1"
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        os.environ['https_proxy'] = "https://172.16.1.2:3128"
        os.environ['HTTPS_PROXY'] = "https://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        #taskname=args.traintag #"eli5asksciencemodeling"
    else:
        trainoutput=args.outputdir #"./output"
        #taskname=args.traintag #taskname="eli5asksciencemodeling"
        mycache_dir="./data/"
    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    raw_datasets, text_column, target_column, task_column = loaddata(args, USE_HPC)
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
    
    model, feature_extractor, starting_epoch = loadmodel(model_checkpoint, id2label, label2id, task=task, mycache_dir=mycache_dir, 
                                                         pretrained=args.pretrained, hpc=USE_HPC, unfreezename=args.unfreezename, 
                                                         return_attention_mask=True, freeze_feature_encoder=True, ignore_mismatched_sizes=False)
    model_input_name = feature_extractor.model_input_names[0]
    print("model_input_name:", model_input_name) #input_values
    
    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        task_column, features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    sample = raw_datasets["train"][0]["audio"]
    print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    print(f"inputs keys: {list(inputs.keys())}")
    print(
        f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
    )

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples[task_column]] #"audio"
        target_labels = [np.int64(x) for x in examples[target_column]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * args.max_length_seconds),
            truncation=True,
            return_attention_mask=True,
        )
        #return inputs
        output_batch = inputs #{model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = target_labels #list(target_labels) #list(examples[target_column])
        return output_batch
    
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
            inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[target_column])
        else:
            audio = batch[task_column]
            wav = random_subsample(
                    audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
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
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[target_column])

        return output_batch

    # Set the training transforms
    #raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)
    #transferred_datasets = DatasetDict()
    #transferred_datasets["train"]=raw_datasets["train"].map(train_transforms)
    # # Set the validation transforms
    # raw_datasets[valkey].set_transform(val_transforms, output_all_columns=False)
    # #transferred_datasets[valkey]=raw_datasets[valkey].map(val_transforms)
    # train_dataset=raw_datasets["train"]
    # eval_dataset=raw_datasets[valkey]
    # #train_dataset=transferred_datasets["train"]
    # #eval_dataset=transferred_datasets[valkey]

    dataset_encoded = raw_datasets.map(
        preprocess_function,
        remove_columns=raw_datasets["train"].column_names, #["audio", "file"],
        batched=True,
        batch_size=100,
        num_proc=1,
    )
    #dataset_encoded = dataset_encoded.rename_column("genre", "label")


    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    training_args = TrainingArguments(
        trainoutput,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.total_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        push_to_hub=False,
    )
    
    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded[valkey],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    if args.training == True:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


    

