from datasets import load_dataset, DatasetDict, features, load_metric
import evaluate
import torch
import os
import numpy as np
from sklearn.metrics import classification_report
from transformers import EvalPrediction
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path

class myEvaluator:
    def __init__(self, task, useHFevaluator=False, dualevaluator=False, labels=None, processor=None, mycache_dir=None, output_path=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.output_path = output_path
        self.task = task
        self.preds = []
        self.refs = []
        self.labels = labels
        self.processor = processor
        self.HFmetric = None
        if self.task == "audio-classification":
            self.metricname = "accuracy" #"mse" "wer"
        elif self.task == "audio-asr":
            self.metricname = "wer" #word error rate (WER)
        self.LOmetric = None
        #https://huggingface.co/spaces/evaluate-metric/wer
        #if self.task.startswith("audio"):
        if self.useHFevaluator:
            # Load the accuracy metric from the datasets package
            self.HFmetric = evaluate.load(self.metricname, cache_dir=mycache_dir) #evaluate.load("mse")
        elif self.metricname == "wer":
            self.LOmetric = load_metric("wer", trust_remote_code=True)#deprecated

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    #eval_pred is EvalPrediction type
    def compute_metrics(self, eval_pred: EvalPrediction):
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions #(1000, 593, 46)
        label_ids = eval_pred.label_ids #(462, 79) (1000, 109)
        if self.metricname == "accuracy":
            """Computes accuracy on a batch of predictions"""
            preds = np.argmax(preds, axis=1)
            #return self.HFmetric.compute(predictions=preds, references=label_ids)
        elif self.metricname == "mse":
            preds = np.squeeze(preds)
            #return self.HFmetric.compute(predictions=preds, references=label_ids)
        elif self.metricname == "wer":
            preds = np.argmax(preds, axis=-1)#(462, 329, 32)->(462, 329)
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            preds = self.processor.batch_decode(preds)#get str
            # we do not want to group tokens when computing the metrics
            label_ids = self.processor.batch_decode(label_ids, group_tokens=False)#get str (462,79)->(462)

        return self.compute(predictions=preds, references=label_ids)

    def mycompute(self, predictions=None, references=None):
        predictions = np.array(predictions)
        references = np.array(references)
        if self.metricname == "accuracy":
            eval_result = (predictions == references).astype(np.float32).mean().item()
            # if self.labels:
            #     print("Classification report", classification_report(references, predictions, target_names=self.labels))
        elif self.metricname == "mse": #mse
            eval_result = ((predictions - references) ** 2).mean().item()
        elif self.metricname == "wer":
            eval_result = self.LOmetric.compute(predictions=predictions, references=references)
        results = {self.metricname: eval_result}
        return results
    
    def compute(self, predictions=None, references=None):
        results = {}
        if predictions is not None and references is not None:
            if self.useHFevaluator:
                results = self.HFmetric.compute(predictions=predictions, references=references)
            else: 
                results = self.mycompute(predictions=predictions, references=references)
            #print("HF evaluator:", results)
            self.create_df(predictions, references, display=True)
            if not isinstance(results, dict):
                #wer output is float, convert to dict
                results = {self.metricname: results}
        else: #evaluate the whole dataset
            if self.useHFevaluator:
                results = self.HFmetric.compute()
                print("HF evaluator result1:", results)
                results2 = self.HFmetric.compute(predictions=self.preds, references=self.refs) #the same results
                print("HF evaluator result2:", results2)
                if not isinstance(results, dict):
                    #wer output is float, convert to dict
                    results = {self.metricname: results}
            else:
                results = self.mycompute(predictions=self.preds, references=self.refs)
            self.create_df(self.preds, self.refs, display=True)
            self.preds.clear()
            self.refs.clear()
        return results
    
    def create_df(self, predictions, references, display=True):
        # dictionary of lists 
        resultdict = {'predictions': predictions, 'references': references} 
        df = pd.DataFrame(resultdict)
        if display:
            #print(df.head())
            print("Total len:", len(df))
            print(df.sample(n=8))
        if self.output_path:
            filepath = Path(os.path.join(self.output_path, 'evaluationresult.csv'))  
            if filepath.exists():
                df.to_csv(filepath, mode='a', header=False) #append to existing csv file
            else:#create a new csv file:
                filepath.parent.mkdir(parents=True, exist_ok=True)  
                #df.to_csv(filepath)
                df.to_csv(filepath, mode='a', header=True)
            print("Evaluation dataframe saved to:", str(filepath))

    def add_batch(self, predictions, references):
        if self.useHFevaluator == True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        #self.preds.append(predictions)
        self.refs.extend(references)
        self.preds.extend(predictions)
        #references: list of list
        # for ref in references:
        #     self.refs.append(ref[0])
        #print(len(self.refs))

def evaluate_dataset(model, eval_dataloader, device, metric, processor=None):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    model.eval()
    for batch in tqdm(eval_dataloader):#'input_values' [64, 16000]
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        logits = outputs.logits #[4, 12]
        #Get the class with the highest probability
        logitsmax=torch.argmax(logits, dim=-1) #argmax (https://pytorch.org/docs/stable/generated/torch.argmax.html) for the last dimension, torch.Size([4])
        #predicted_class_ids = logitsmax.item() #.item() moves the data to CPU
        
        if metric.metricname == "wer": #audio-asr
            label_ids = batch["labels"]
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            decoded_preds = processor.batch_decode(logitsmax)
            decoded_labels = processor.batch_decode(label_ids, group_tokens=False)
        else:
            #use the model’s id2label mapping to convert it to a label:
            #decoded_preds = [model.config.id2label[str(pred.item())] for pred in logitsmax]
            decoded_preds = [pred.item() for pred in logitsmax]
            #result = model.config.id2label[str(predicted_class_ids)]
            labels = batch["labels"] #label is also integer
            decoded_labels = [label.item() for label in labels]

        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # result1 = metric.compute(predictions=decoded_preds, references=decoded_labels)
        #evalmetric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    results = metric.compute()
    #evalresults = evalmetric.compute()
    print(f"Evaluation score: {results}")
    #print(evalresults['score'])
    return results
