from datasets import load_dataset
from datasets import load_metric
import torch
import json
from pathlib import Path
import os
from transformers import pipeline
from transformers import DistilBertTokenizerFast, AutoTokenizer
from transformers import DistilBertForQuestionAnswering, AutoModelForQuestionAnswering
from transformers import get_scheduler
from transformers import pipeline
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
import collections
import numpy as np
import evaluate
from transformers import AutoTokenizer
import transformers
import torch
from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer, AutoConfig, AutoModel
def loadmodels(model_ckpt, mycache_dir, newname):
    #model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)#,cache_dir=mycache_dir)
    config = AutoConfig.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    newpath=os.path.join(mycache_dir, newname)
    tokenizer.save_pretrained(newpath)
    config.save_pretrained(newpath)
    model.save_pretrained(newpath)
    print(model)
#loadmodels("meta-llama/Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf")

#https://huggingface.co/meta-llama/Meta-Llama-3-8B
def LLMmodel(modelname="meta-llama/Llama-2-7b-chat-hf"):

    model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto")
    model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")
    # By default, the output will contain up to 20 tokens
    # Setting `max_new_tokens` allows you to control the maximum length
    generated_ids = model.generate(**model_inputs, max_new_tokens=50)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    model_inputs = tokenizer("Where is San Jose, CA?", return_tensors="pt").to("cuda")
    del model_inputs["token_type_ids"]
    generated_ids = model.generate(**model_inputs, max_new_tokens=150)
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result

def LLMmodel_quant(modelname="meta-llama/Llama-2-7b-chat-hf"):
    from transformers import BitsAndBytesConfig #pip install bitsandbytes
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )#requires CUDA

    #model_nf4 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=nf4_config)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
            device_map='auto',
            load_in_8bit=True,
            max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')
    
    model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

    # LLM + greedy decoding = repetitive, boring output
    generated_ids = model.generate(**model_inputs)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # With sampling, the output becomes more creative!
    generated_ids = model.generate(**model_inputs, do_sample=True)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def LLMinference(modelname):
    modelname = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    pipeline = transformers.pipeline(
        "text-generation",
        model=modelname,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    sequences = pipeline(
        'Where is San Jose, CA? \n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def QAinference(model, tokenizer, question, context, device, usepipeline=True):
    if usepipeline ==True:
        if device.type == 'cuda':
            question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0) #device=0 means cuda
        else:
            question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer) 
        answers=question_answerer(question=question, context=context)
        print(answers) #'answer', 'score', 'start', 'end'
    else:
        inputs = tokenizer(question, context, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        #Get the highest probability from the model output for the start and end positions:
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        #predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        #Decode the predicted tokens to get the answer:
        predict_answer_tokens = inputs['input_ids'][0, answer_start_index : answer_end_index + 1]
        answers=tokenizer.decode(predict_answer_tokens)
        print(answers)
    return answers

def get_device(gpuid='0', useamp=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        useamp = False
    else:
        device = torch.device("cpu")
        useamp = False
    print('Using device:', device)
    return device, useamp

def load_seqmodel(model_name_or_path, task='chat', device = "auto", dtype = torch.bfloat16, load_only=True, labels=None, mycache_dir=None, trust_remote_code=True):
    tokenizer = None
    model = None
    if task == 'QA' and 'distilbert' in model_name_or_path:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name_or_path)
        #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = DistilBertForQuestionAnswering.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)
        #model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint) #"distilbert-base-uncased")
        #Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)

    return model, tokenizer

class MultitaskSeq():
    def __init__(self, model_name, model_path="", model_type="huggingface", task="QA", cache_dir="./output", gpuid='0') -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        if isinstance(model_name, str) and model_type=="huggingface":
            #os.environ['HF_HOME'] = cache_dir #'~/.cache/huggingface/'
            if model_path and os.path.exists(model_path):
                model_name_or_path = model_path
            else:
                model_name_or_path = model_name
            self.model, self.tokenizer = load_seqmodel(model_name_or_path = model_name_or_path, task=task, device = self.device)
        #self.model=self.model.to(self.device)
        self.model.eval()

    def __call__(self, text, context=None, max_new_tokens=100):
        #text is a list
        if self.task == 'QA':
            result=QAinference(self.model, self.tokenizer, text, context, self.device, usepipeline=False) #not correct before training {'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}{'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}
        elif self.task == 'text-generation' or self.task == 'chat':
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            # By default, the output will contain up to 20 tokens
            # Setting `max_new_tokens` allows you to control the maximum length
            generated_ids = self.model.generate(**model_inputs, \
                                                max_new_tokens=max_new_tokens, \
                                                pad_token_id=self.tokenizer.eos_token_id)
            result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif self.task == 'chat-template':
            input_ids = self.tokenizer.apply_chat_template(text, return_tensors="pt").to(self.device)
            prompt_len = input_ids.shape[-1]
            print("prompt_len:", prompt_len)
            output = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, pad_token_id=0)
            result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return result

def test_llama():
    #"meta-llama/Llama-2-7b-chat-hf"
    model_id = "meta-llama/Meta-Llama-3-8B" #"meta-llama/Llama-2-7b-chat-hf"
    multitaskseq=MultitaskSeq(model_name=model_id, task="chat")
    results=multitaskseq(['Where is San Jose, CA? \n'], max_new_tokens=300)
    print(results)

def test_llama_guard():
    #https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B
    model_id = "meta-llama/Meta-Llama-Guard-2-8B"
    multitaskseq=MultitaskSeq(model_name=model_id, task="chat")
    results=multitaskseq([
        {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
        {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
    ])
    print(results) #'safe'

#Test QA
    # 
    #  answers=QAinference(model, tokenizer, question, context, device, usepipeline=False) #not correct before training {'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}{'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}
def test_QA():
    multitaskseq=MultitaskSeq(model_name='distilbert-base-uncased', task="QA")
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    results=multitaskseq(text=question, context=context)
    print(results)

#https://huggingface.co/docs/transformers/main/conversations
def test_chat():
    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]
    pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    response = pipe(chat, max_new_tokens=512)
    print(response[0]['generated_text'][-1]['content'])

    #You can continue the chat by appending your own response to it. 
    # The response object returned by the pipeline actually contains the entire chat so far, 
    # so we can simply append a message and pass it back:
    chat = response[0]['generated_text']
    chat.append(
        {"role": "user", "content": "Wait, what's so wild about soup cans?"}
    )
    response = pipe(chat, max_new_tokens=512)
    print(response[0]['generated_text'][-1]['content'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple HFseq model inference')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--model_checkpoint', type=str, default="distilbert-base-uncased",
                    help='Model checkpoint name from https://huggingface.co/models, "bert-base-cased"')
    parser.add_argument('--task', type=str, default="QA",
                    help='NLP tasks: QA, sentiment, token_classifier, sequence_classifier, chat')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    args = parser.parse_args()

    #test_llama_guard()
    #test_QA()
    test_llama()

    # global task
    # task = args.task
    # model_checkpoint = args.model_checkpoint
    # global tokenizer
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model.to(device)

    