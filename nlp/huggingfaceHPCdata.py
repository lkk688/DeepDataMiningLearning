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
from datasets import load_dataset
datasetpath='emotion/split/1.0.0/cca5efe2dfeb58c1d098e0f9eeb200e9927d889b5a03c67097275dfb5fe463bd'
trainarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-train.arrow')
valarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-validation.arrow')
testarrowpath=os.path.join(mycache_dir, datasetpath, 'emotion-test.arrow')
dataset = load_dataset("arrow", data_files={'train': trainarrowpath, 'val': valarrowpath, 'test': testarrowpath})

print(dataset)
train_ds = dataset["train"]
print(train_ds)

# dataset = load_dataset('imdb', cache_dir=mycache_dir)

# emotions = load_dataset("emotion")
# train_ds = emotions["train"]
# print(len(train_ds))
# print(train_ds.column_names)


# from transformers import AutoTokenizer

# model_ckpt = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)

# from transformers import AutoModel
# model = AutoModel.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)