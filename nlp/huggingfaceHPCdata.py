#export HF_DATASETS_CACHE="/path/to/another/directory"
# export HF_HOME=\my_drive\hf\misc
# export HF_DATASETS_CACHE=\my_drive\hf\datasets
# export TRANSFORMERS_CACHE=\my_drive\hf\models

import os
mycache_dir="/data/cmpe249-fa23/Huggingfacecache"
os.environ['TRANSFORMERS_CACHE'] = mycache_dir
os.environ['HF_HOME'] = mycache_dir
os.environ['HF_DATASETS_CACHE'] = mycache_dir

import torch
print(torch.__version__)

#https://huggingface.co/docs/datasets/loading
from datasets import load_dataset, ReadInstruction
datasetpath='emotion/split/1.0.0/cca5efe2dfeb58c1d098e0f9eeb200e9927d889b5a03c67097275dfb5fe463bd'
trainarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-train.arrow')
valarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-validation.arrow')
testarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-test.arrow')
dataset = load_dataset("arrow", data_files={'train': trainarrowpath, 'val': valarrowpath, 'test': testarrowpath})

print(dataset)
train_ds = dataset["train"]
print(train_ds)

#eli5 = load_dataset("eli5", split="train_asks")
eli5 = load_dataset("eli5")
print(eli5)

imdb_dataset = load_dataset("imdb")
imdb_dataset
# dataset = load_dataset('imdb', cache_dir=mycache_dir)

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")

#https://huggingface.co/datasets/opus100
raw_datasets = load_dataset("opus100", language_pair="en-zh")

#https://huggingface.co/datasets/wmt19
raw_datasets = load_dataset("wmt19", language_pair=("zh","en"))

# emotions = load_dataset("emotion")
# train_ds = emotions["train"]
# print(len(train_ds))
# print(train_ds.column_names)

# train_ds, test_ds = load_dataset('bookcorpus', split=[
#     ReadInstruction('train'),
#     ReadInstruction('test'),
# ])
#train_ds, test_ds = load_dataset('bookcorpus', split=['train', 'test'])
train_ds = load_dataset('bookcorpus')
from transformers import AutoTokenizer, AutoConfig, AutoModel

def loadmodels(model_ckpt):
    #model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)
    config = AutoConfig.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    newpath=os.path.join(mycache_dir, model_ckpt)
    tokenizer.save_pretrained(newpath)
    config.save_pretrained(newpath)
    model.save_pretrained(newpath)
    print(model)

loadmodels("distilbert-base-uncased")

loadmodels("distilroberta-base")

loadmodels("distilgpt2")

loadmodels("gpt2")

loadmodels("meta-llama/Llama-2-7b-chat-hf")
# model_ckpt = "distilroberta-base"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)
# config = AutoConfig.from_pretrained(model_ckpt)
# newpath=os.path.join(mycache_dir, model_ckpt)
# tokenizer.save_pretrained(newpath)
# config.save_pretrained(newpath)
# model.save_pretrained(newpath)

print("Done")
# from transformers import AutoModel
# model = AutoModel.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)