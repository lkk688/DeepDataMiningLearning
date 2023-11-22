from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import torch
import os
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math
import collections
import numpy as np

import os
global data_field
data_field='answers.text'
global block_size
block_size = 512 #128
valkey="test"#"validation"
global globaltokenizer
global source_lang
source_lang="en"
global target_lang
target_lang="zh"#"fr"

#https://huggingface.co/facebook/wmt21-dense-24-wide-en-x
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
# tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

# inputs = tokenizer("To translate into a target language, the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the forced_bos_token_id parameter to the generate method.", return_tensors="pt")

# # translate English to Chinese
# generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("zh")) #max_new_tokens
# result=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(result)

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

# from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# from transformers import AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def loadmodel(model_checkpoint, task="Seq2SeqLM", mycache_dir="", pretrained="", hpc=True):
    if hpc==True:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(localpath)
        if task=="Seq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(localpath)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(localpath)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)#, cache_dir=mycache_dir)
        if task=="Seq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)#"distilroberta-base")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer

max_length = 128
def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]] #1000
    targets = [ex[target_lang] for ex in examples["translation"]] #1000
    model_inputs = globaltokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

def loaddata(args, USE_HPC):
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
                data_field='text'
                #source_lang="en"
                #global target_lang
                target_lang="fr"
                #sampletext = "This is a great [MASK]."
            elif args.data_name=='opus100':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'enzh')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                valarrowpath=os.path.join(datasetpath, args.data_name+'-validation.arrow')
                testarrowpath=os.path.join(datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                #data_field='text'
                #source_lang="en"
                #global target_lang
                target_lang="zh"
                #raw_datasets = load_dataset("opus100", language_pair="en-zh")
        else:
            if args.data_name=='kde4':
                raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
            elif args.data_name=='opus100':
                raw_datasets = load_dataset("opus100", language_pair="en-zh")
            else:
                raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
        #Download to home/.cache/huggingface/dataset
        
        print("All keys in raw datasets:", raw_datasets['train'][0].keys())
        split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
        # rename the "test" key to "validation" 
        #split_datasets["validation"] = split_datasets.pop("test")
        #one element
        sampletext = split_datasets["train"][1]["translation"]
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
            raw_datasets["test"] = raw_datasets["test"].shuffle(seed=42).select([i for i in list(range(testlen))])
    
    return raw_datasets

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

#data_name="imdb", dataconfig="", model_checkpoint="distilbert-base-uncased"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="opus100",
                    help='data name: kde4')
    parser.add_argument('--dataconfig', type=str, default='',
                    help='train_asks[:5000]')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache",
                    help='path to get data ') #r"E:\Dataset\NLPdataset\aclImdb"
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wmt21-dense-24-wide-en-x",
                    help='Model checkpoint name from h ttps://huggingface.co/models, Helsinki-NLP/opus-mt-en-fr')
    parser.add_argument('--task', type=str, default="Seq2SeqLM",
                    help='NLP tasks: ')
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="1121nohfacc",
                    help='Name the current training')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--usehpc', type=bool, default=False,
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', type=bool, default=False,
                    help='Use Huggingface accelerator')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    use_accelerator = args.useHFaccelerator
    model_checkpoint = args.model_checkpoint
    
    USE_HPC=args.usehpc
    if USE_HPC:
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
        os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        os.environ['HF_EVALUATE_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        #os.environ['https_proxy'] = "https://172.16.1.2:3128"
        #os.environ['HTTPS_PROXY'] = "https://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        taskname=args.traintag #"eli5asksciencemodeling"
    else:
        trainoutput=args.outputdir #"./output"
        taskname=args.traintag #taskname="eli5asksciencemodeling"
        mycache_dir="./data/"
    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    model, tokenizer = loadmodel(model_checkpoint, task=task, mycache_dir=mycache_dir, pretrained=args.pretrained, hpc=USE_HPC)
    tokenizer.model_max_len=512
    print(tokenizer.pad_token)
    print(tokenizer.eos_token)
    globaltokenizer = tokenizer
    #tokenizer.pad_token = tokenizer.eos_token

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")
    modelparameters(model, args.unfreezename)

    raw_datasets = loaddata(args, USE_HPC)
    if tokenizer:
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_datasets["train"].column_names,
        )#The default batch size is 1000, but you can adjust it with the batch_size argument
        tokenized_datasets.set_format("torch")
    else:
        tokenized_datasets = {}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    #To test this on a few samples
    batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
    print(batch.keys()) #(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids'])
    #batch["labels"] #our labels have been padded to the maximum length of the batch, using -100:
    #batch["decoder_input_ids"] #shifted versions of the labels

    metric = evaluate.load("sacrebleu") #pip install sacrebleu
    predictions = [
        "This plugin lets you translate web pages between several languages automatically."
    ]
    references = [
        [
            "This plugin allows you to automatically translate web pages between several languages."
        ]
    ]
    metric.compute(predictions=predictions, references=references)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets[valkey], collate_fn=data_collator, batch_size=8
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_train_epochs = args.total_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if use_accelerator:
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
        device = accelerator.device
        print("Using HF Accelerator and device:", device)
    else:
        device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
        model.to(device)
        print("Using device:", device)
    
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            if not use_accelerator:
                batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not use_accelerator:
                loss.backward()
            else:
                accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                if not use_accelerator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    generated_tokens = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=128,
                    )
                else:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=128,
                    )
            labels = batch["labels"]
            if use_accelerator:
                # Necessary to pad predictions and labels for being gathered
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(generated_tokens)
                labels_gathered = accelerator.gather(labels)

                decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
            else:
                decoded_preds, decoded_labels = postprocess(generated_tokens, labels)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        results = metric.compute()
        print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    del model, optimizer, lr_scheduler
    if use_accelerator:
        accelerator.free_memory()