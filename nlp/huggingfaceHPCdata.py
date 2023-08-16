#export HF_DATASETS_CACHE="/path/to/another/directory"
# export HF_HOME=\my_drive\hf\misc
# export HF_DATASETS_CACHE=\my_drive\hf\datasets
# export TRANSFORMERS_CACHE=\my_drive\hf\models

import os
os.environ['TRANSFORMERS_CACHE'] = "/data/cmpe249-fa22/Huggingfacecache"
os.environ['HF_HOME'] = "/data/cmpe249-fa22/Huggingfacecache"
os.environ['HF_DATASETS_CACHE'] = "/data/cmpe249-fa22/Huggingfacecache"

mycache_dir="/data/cmpe249-fa22/Huggingfacecache"
from datasets import load_dataset

dataset = load_dataset('imdb', cache_dir=mycache_dir)

emotions = load_dataset("emotion")
train_ds = emotions["train"]
print(len(train_ds))
print(train_ds.column_names)


from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)

from transformers import AutoModel
model = AutoModel.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)