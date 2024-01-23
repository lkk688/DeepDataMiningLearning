import logging
import os
import torch

valkey='test'
TrustRemoteCode=True

logger = logging.getLogger(__name__)

def get_device(gpuid='0', useamp=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        useamp = False
    else:
        device = torch.device("cpu")
        useamp = False
    return device, useamp

def deviceenv_set(USE_HPC, data_path=""):
    if USE_HPC:
        #https://huggingface.co/docs/transformers/installation#offline-mode
        #HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        mycache_dir=data_path #"/data/cmpe249-fa23/Huggingfacecache"
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
        if os.path.exists(data_path):
            mycache_dir=data_path
            os.environ['HF_HOME'] = mycache_dir
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        elif os.environ.get('HF_HOME') is not None:
            mycache_dir=os.environ['HF_HOME']
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        else:
            mycache_dir="./data/"
            os.environ['HF_HOME'] = mycache_dir
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # mycache_dir=os.path.join('D:',os.sep, 'Cache','huggingface')
        
        print("HF_HOME:", os.environ['HF_HOME'])
        print("HF_DATASETS_CACHE:", os.environ['HF_DATASETS_CACHE'])
        # os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # if os.environ.get('HF_HOME') is None:
        #     mycache_dir=args.data_path
        #     os.environ['HF_HOME'] = mycache_dir
        #     os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # else:
        #     print("HF_HOME:", os.environ['HF_HOME'])
        #     mycache_dir=os.environ['HF_HOME']
        #trainoutput=args.outputdir #"./output"
        #taskname=args.traintag #taskname="eli5asksciencemodeling"
    
    return mycache_dir