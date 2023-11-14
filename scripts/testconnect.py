import os
#export https_proxy="https://172.16.1.2:3128"
#export http_proxy="http://172.16.1.2:3128"
os.environ['http_proxy'] = "http://172.16.1.2:3128"
os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
os.environ['https_proxy'] = "" #"https://172.16.1.2:3128"
os.environ['HTTPS_PROXY'] = "" #"https://172.16.1.2:3128"

import urllib3

proxy = urllib3.ProxyManager("http://172.16.1.2:3128")#https does not work
proxy.request("GET", "https://google.com/")

import requests

proxies = {"http": "http://172.16.1.2:3128", "https": "http://172.16.1.2:3128"}
r = requests.get("https://www.google.com/", proxies=proxies, verify=False)
print(r)


mycache_dir="/data/cmpe249-fa23/Huggingfacecache"
os.environ['TRANSFORMERS_CACHE'] = mycache_dir
os.environ['HF_HOME'] = mycache_dir
os.environ['HF_DATASETS_CACHE'] = mycache_dir
#Does not work
#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir=mycache_dir)