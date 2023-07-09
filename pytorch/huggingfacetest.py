import torch
from tqdm.auto import tqdm
import numpy as np
import evaluate
import os
import pandas as pd

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

    text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
    from your online store in Germany. Unfortunately, when I opened the package, \
    I discovered to my horror that I had been sent an action figure of Megatron \
    instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
    dilemma. To resolve the issue, I demand an exchange of Megatron for the \
    Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
    this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
    classifier = pipeline("text-classification")
    outputs = classifier(text)
    print(pd.DataFrame(outputs))

    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    print(pd.DataFrame(outputs))

    reader = pipeline("question-answering")
    question = "What does the customer want?"
    outputs = reader(question=question, context=text)
    print(pd.DataFrame([outputs]))

    summarizer = pipeline("summarization")
    outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
    print(outputs[0]['summary_text'])

    translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    print(outputs[0]['translation_text'])

    generator = pipeline("text-generation")
    response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
    prompt = text + "\n\nCustomer service response:\n" + response
    outputs = generator(prompt, max_length=200)
    print(outputs[0]['generated_text'])

    #pip install xformers
    #pip3 install emoji==0.6.0