from datasets import load_dataset
from datasets import load_metric
import torch
import json
from pathlib import Path
import os
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="squad",
                    help='data name: imdb, conll2003, "glue", "mrpc" ')
    parser.add_argument('--data_path', type=str, default=r"E:\Dataset\NLPdataset\squad",
                    help='path to get data')
    parser.add_argument('--model_checkpoint', type=str, default="distilbert-base-uncased",
                    help='Model checkpoint name from https://huggingface.co/models, "bert-base-cased"')
    parser.add_argument('--task', type=str, default="QA",
                    help='NLP tasks: sentiment, token_classifier, "sequence_classifier"')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task
    model_checkpoint = args.model_checkpoint
    global tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_checkpoint)
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = DistilBertForQuestionAnswering.from_pretrained(model_checkpoint)
    #model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint) #"distilbert-base-uncased")
    #Some weights of DistilBertForQuestionAnswering were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #Test QA
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    answers=QAinference(model, tokenizer, question, context, device, usepipeline=False) #not correct before training {'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}{'score': 0.004092414863407612, 'start': 14, 'end': 57, 'answer': 'billion parameters and can generate text in'}
