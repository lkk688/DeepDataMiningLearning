#https://huggingface.co/transformers/v4.1.1/custom_datasets.html
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
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="custom",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="imdb",
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
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
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
    #model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    #Test QA
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)
    print(question_answerer(question=question, context=context))

    train_contexts, train_questions, train_answers = read_squad(os.path.join(args.data_path, 'train-v2.0.json'))
    val_contexts, val_questions, val_answers = read_squad(os.path.join(args.data_path, 'dev-v2.0.json'))

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.learningrate)

    num_epochs = args.total_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            #batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            #outputs = model(**batch)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            #sequence classification: outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss
            loss = outputs[0]
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.update(1)
    
    outputpath=os.path.join(args.outputdir, task, args.data_name)
    tokenizer.save_pretrained(outputpath)
    torch.save(model.state_dict(), os.path.join(outputpath, 'savedmodel.pth'))
    
    model.eval()