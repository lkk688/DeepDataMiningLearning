import torch
from tqdm.auto import tqdm
import numpy as np
import evaluate
import os

from transformers import pipeline
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import AdamW
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_name', type=str, default="conll2003",
                    help='data name: conll2003, "glue", "mrpc" ')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa22/ImageClassData",
                    help='path to get data')
    parser.add_argument('--model_checkpoint', type=str, default="xlm-roberta-large-finetuned-conll03-english",
                    help='Model checkpoint name from https://huggingface.co/models')
    parser.add_argument('--task', type=str, default="token_classifier",
                    help='NLP tasks: "sequence_classifier"')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task

    model_checkpoint = args.model_checkpoint

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

    classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple") #"ner"
    result1 = classifier("Alya told Jasmine that Andrew could pay with cash..")
    print(result1)

    # token_classifier = pipeline(
    #     "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    # )
    result2 = classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(result2)

    sentiment_pipeline = pipeline("sentiment-analysis")
    data = ["I love you", "I hate you"]
    print(sentiment_pipeline(data))

    specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    print(specific_model(data))

    #pip install xformers
    #pip3 install emoji==0.6.0