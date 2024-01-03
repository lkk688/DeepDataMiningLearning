from huggingface_hub import model_info; 
print(model_info('gpt2'))

import os
#hfhome_dir=os.path.join('D:\\','Cache','huggingface')#"/data/cmpe249-fa23/Huggingfacecache"
hfhome_dir=os.path.join('D:',os.sep, 'Cache','huggingface')
#os.environ['TRANSFORMERS_CACHE'] = hfhome_dir
os.environ['HF_HOME'] = hfhome_dir
#os.environ['HF_HUB_CACHE'] = os.path.join(hfhome_dir, 'hub')
#os.environ['HF_DATASETS_CACHE'] = hfhome_dir
#HF_HUB_OFFLINE=1

import evaluate
metric = evaluate.load("sacrebleu") #pip install sacrebleu
metric = evaluate.load("accuracy") #save to /data/cmpe249-fa23/Huggingfacecache/metrics
metric = evaluate.load("squad")

from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
print(imdb_dataset)