#huggingfaceSequence4: created on 11/26, add QA
#Sequence3:created on 11/24, plan to add summarization
from datasets import load_dataset, DatasetDict
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator, EvalPrediction)
import evaluate
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import math
import collections
import numpy as np
import random
import json
import os
valkey="test"#"validation"
#Dualevaluation=True
from utils_qa import postprocess_qa_predictions, create_and_fill_np_array, updateQAtraininputs, updateQAvalinputs


#https://huggingface.co/facebook/wmt21-dense-24-wide-en-x
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
# tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

# inputs = tokenizer("To translate into a target language, the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the forced_bos_token_id parameter to the generate method.", return_tensors="pt")

# # translate English to Chinese
# generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("zh")) #max_new_tokens
# result=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(result)

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

# from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# from transformers import AutoModelForSeq2SeqLM
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

from mybertmodel import load_QAbertmodel
def loadmodel(model_checkpoint, task="QA", mycache_dir="", pretrained="", hpc=True, unfreezename=""):
    if model_checkpoint.startswith('my'):
        model, tokenizer = load_QAbertmodel()
    elif hpc==True:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(localpath, local_files_only=True)
        if task in ['translation', 'summarization', 'Seq2SeqLM']:
            model = AutoModelForSeq2SeqLM.from_pretrained(localpath, local_files_only=True)
        elif task in ['QA', 'qa', 'QuestionAnswering']:
            model = AutoModelForQuestionAnswering.from_pretrained(localpath, local_files_only=True)
            #model = DistilBertForQuestionAnswering.from_pretrained(model_checkpoint)
        else:
            model = AutoModel.from_pretrained(localpath, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)#, cache_dir=mycache_dir)
        if task in ['translation', 'summarization', 'Seq2SeqLM']:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        elif task in ['QA', 'qa', 'QuestionAnswering']:
            model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
            #model = DistilBertForQuestionAnswering.from_pretrained(model_checkpoint)
        else:
            model = AutoModel.from_pretrained(model_checkpoint)
    starting_epoch = 0
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        starting_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['model_state_dict'])
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print("Embeeding size:", embedding_size) #65001
    print("Tokenizer length:", len(tokenizer)) #65001
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")
    modelparameters(model, unfreezename)
    return model, tokenizer, starting_epoch

# max_length = 128
# def preprocess_function(examples):
#     inputs = [ex[source_lang] for ex in examples["translation"]] #1000
#     targets = [ex[target_lang] for ex in examples["translation"]] #1000
#     model_inputs = globaltokenizer(
#         inputs, text_target=targets, max_length=max_length, truncation=True
#     )
#     return model_inputs

def loaddata(args, USE_HPC):
    task_column =""
    text_column = ""
    target_column =""
    if args.data_type == "huggingface":
        if USE_HPC:
            if args.data_name=='kde4':
                #raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
                datasetpath=os.path.join(mycache_dir, args.data_name, "en-fr-lang1\=en\,lang2\=fr", "0.0.0", "/243129fb2398d5b0b4f7f6831ab27ad84774b7ce374cf10f60f6e1ff331648ac") #"/data/cmpe249-fa23/Huggingfacecache/imdb/plain_text/1.0.0/d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0"
                #raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir) #eli5
                datasetpath=os.path.join(mycache_dir, args.data_name)
                trainarrowpath=os.path.join(mycache_dir, args.data_name, args.data_name+'-train.arrow')
                #valarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-validation.arrow')
                #testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "en"
                target_column = "fr"
            elif args.data_name=='opus100':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'enzh')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                valarrowpath=os.path.join(datasetpath, args.data_name+'-validation.arrow')
                testarrowpath=os.path.join(datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                #raw_datasets = load_dataset("opus100", language_pair="en-zh")
                text_column =  "en"
                target_column = "zh"
            elif args.data_name == 'wmt19':
                datasetpath=os.path.join(mycache_dir, args.data_name, 'zh-en-24b9c423f6ba2174/0.0.0/29e210fae5690e843cae5dc43b53db36c4e02f927db50cd5235a22ab42dde90a')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train*.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "en"
                target_column = "zh"
            elif args.data_name=='billsum': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, 'default/3.0.0/75cf1719d38d6553aa0e0714c393c74579b083ae6e164b2543684e3e92e0c4cc')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column = "text"
                target_column = "summary"
            elif args.data_name=='cnn_dailymail': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, '3.0.0/3.0.0/1b3c71476f6d152c31c1730e83ccb08bcf23e348233f4fcc11e182248e6bf7de')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train*.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column =  "article" #("article", "highlights")
                target_column = "highlights"
            elif args.data_name=='xsum': #summarization
                datasetpath=os.path.join(mycache_dir, args.data_name, 'default/1.2.0/082863bf4754ee058a5b6f6525d0cb2b18eadb62c7b370b095d1364050a52b71')
                trainarrowpath=os.path.join(datasetpath, args.data_name+'-train.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath})
                text_column = "document"
                target_column = "summary"
            else:
                raw_datasets = load_dataset(args.data_name, language_pair=(args.target_lang,args.source_lang))
                text_column =  "en"
                target_column = "zh"
        else:
            if args.data_name=='kde4':
                raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
                task_column ="translation"
                text_column =  "en"
                target_column = "fr"
                #source_lang = args.source_lang.split("_")[0]
                #target_lang = args.target_lang.split("_")[0]
            elif args.data_name=='opus100':
                raw_datasets = load_dataset("opus100", language_pair="en-zh")
                task_column ="translation"
                text_column =  "en"
                target_column = "zh"
            elif args.data_name=='opus_books':
                raw_datasets = load_dataset("opus_books", "en-fr")
                task_column ="translation"
                text_column =  "en"
                target_column = "fr"
            elif args.data_name=='billsum': #summarization
                #raw_datasets = load_dataset("billsum", split="ca_test")
                raw_datasets = load_dataset("billsum")
                text_column = "text"
                target_column = "summary"
            elif args.data_name=='cnn_dailymail': #summarization
                raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
                text_column =  "article" #("article", "highlights")
                target_column = "highlights"
            elif args.data_name=='xsum': #summarization
                raw_datasets = load_dataset("xsum")
                text_column = "document"
                target_column = "summary"
            elif args.data_name=='squad': #QA
                raw_datasets = load_dataset("squad")
                #raw_datasets = load_dataset("squad", split="train[:5000]") #'train', 'test'
                #raw_datasets["train"][0] #'id', 'title','context', 'question', 'answers' (dict with 'text' and 'answer_start'),  
                task_column ="question"
                text_column = "context"
                target_column = "answers"
            else: #wmt19
                #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
                raw_datasets = load_dataset(args.data_name, language_pair=(args.target_lang,args.source_lang))
                text_column = "text"
                target_column = "summary"
        #Download to home/.cache/huggingface/dataset
        
        print("All keys in raw datasets:", raw_datasets['train'][0].keys()) #obly one ['translation'] key
        split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
        # rename the "test" key to "validation" 
        #split_datasets["validation"] = split_datasets.pop("test")
        #one element
        if args.task=="translation":
            sampletext = split_datasets["train"][1][task_column]
            print("Translation text: ", sampletext)
        elif args.task=="summarization":
            sampletext_text = split_datasets["train"][1][text_column]
            sampletext_target = split_datasets["train"][1][target_column]
            print("Summarization context: ", sampletext_text)
            print("Summarization target: ", sampletext_target)
        elif args.task=="QA":
            oneexample = split_datasets["train"][1]
            print("Context: ", oneexample[text_column])
            print("Question: ", oneexample[task_column])
            print("Answer: ", oneexample[target_column])#dict with 'text' and 'answer_start'

        raw_datasets = split_datasets
        if args.subset>0:
            if args.subset<1:
                trainlen=int(len(raw_datasets["train"])*args.subset)
                testlen=int(len(raw_datasets[valkey])*args.subset)
            else:
                trainlen = int(min(args.subset, len(raw_datasets["train"])))
                testlen = int(trainlen/10)
            print("trainlen:", trainlen)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(testlen))])
    
        #limit the evaluation set size
        maxtestlen = 5000
        if len(raw_datasets[valkey])>maxtestlen:
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(maxtestlen))])

    return raw_datasets, text_column, target_column, task_column

def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    #decoded_labels = [[label.strip()] for label in decoded_labels]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def postprocess(predictions, labels, task="translation", ignore_pad_token_for_loss=True):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    if task == "translation":
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
    elif task == "summarization":
        import nltk #pip install nltk
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]
    return decoded_preds, decoded_labels

import sacrebleu #for translation
from rouge_score import rouge_scorer, scoring #for summarization
#ref: https://github.com/huggingface/datasets/blob/main/metrics/rouge/rouge.py
#ref: https://github.com/google-research/google-research/blob/master/rouge/scoring.py
class myRouge:
    def __init__(self, rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=True):
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.use_aggregator = use_aggregator
        if use_aggregator:
            self.aggregator = scoring.BootstrapAggregator()
        else:
            self.scores = []
    
    def _compute(self, predictions, references):
        for ref, pred in zip(references, predictions):
            score = self.scorer.score(ref, pred)
            if self.use_aggregator:
                self.aggregator.add_scores(score)
            else:
                self.scores.append(score)
        
        if self.use_aggregator:
            result = self.aggregator.aggregate()
        else:
            result = {}
            for key in self.scores[0]:
                result[key] = [score[key] for score in self.scores]
        return result



class myEvaluator:
    def __init__(self, args, useHFevaluator=False, dualevaluator=False):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = args.task
        if self.task=="translation":
            self.metricname = "sacrebleu"
            self.language = args.target_lang
        elif self.task=="summarization":
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"] #['rouge1', 'rouge2', 'rougeL']
            self.metricname = "rouge"
        elif self.task in ["qa", "QA", "QuestionAnswering"]:
            self.metricname = "squad"
        
        if useHFevaluator==True or dualevaluator==True:
            self.HFmetric = evaluate.load(self.metricname) #"sacrebleu" pip install sacrebleu
        
        if useHFevaluator==False or dualevaluator==True:
            if self.task=="summarization":
                #self.localscorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
                self.localscorer = myRouge(rouge_types=rouge_types)
        self.preds = []
        self.refs = []

    
    def compute(self, predictions=None, references=None):
        if predictions is not None and references is not None:
            if self.useHFevaluator==True:
                results = self.HFmetric.compute(predictions=predictions, references=references)
                #keys: ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
                print("HF evaluator:", results)
            else:
                if self.task=="translation":
                    bleu = sacrebleu.corpus_bleu(predictions, references)
                    results = {'score':bleu.score, 'counts':bleu.counts, 'totals': bleu.totals,
                            'precisions': bleu.precisions, 'bp': bleu.bp, 
                            'sys_len': bleu.sys_len, 'ref_len': bleu.ref_len
                            }
                elif self.task=="summarization":
                    results = self.localscorer._compute(predictions, references)
                print("Local evaluator:", results)
        else: #evaluate the whole dataset
            if self.useHFevaluator==True or self.dualevaluator==True:
                if self.task == "translation":
                    results = self.HFmetric.compute()
                elif self.task=="summarization":
                    results = self.HFmetric.compute(use_stemmer=True)
                #print("HF evaluator:", results["score"])
                print("HF evaluator:", results)
            
            if self.useHFevaluator==False or self.dualevaluator==True:
                if self.task=="translation":
                    #self.refs should be list of list strings
                    #Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `ja-mecab` for Japanese, `ko-mecab` for Korean and `13a` (mteval) otherwise
                    if self.language=="zh":
                        bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="zh")
                    else:
                        bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="none")
                    results = {'score':bleu.score, 'counts':bleu.counts, 'totals': bleu.totals,
                            'precisions': bleu.precisions, 'bp': bleu.bp, 
                            'sys_len': bleu.sys_len, 'ref_len': bleu.ref_len
                            }
                elif self.task=="summarization":
                    results = self.localscorer._compute(self.preds, self.refs)
                print("Local evaluator:", results)
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator==True or self.dualevaluator==True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        
        if self.useHFevaluator==False or self.dualevaluator==True:
            #self.preds.append(predictions)
            self.refs.extend(references)
            self.preds.extend(predictions)
            #references: list of list
            # for ref in references:
            #     self.refs.append(ref[0])
            #print(len(self.refs))

def evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, num_beams, metric):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    #evalprogress_bar = tqdm(range(num_training_steps))
    model.eval()
    gen_kwargs = {
        "max_length": max_target_length,
        "num_beams": num_beams,
    }
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            if not use_accelerator:
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            #evalprogress_bar.update(1)
        labels = batch["labels"]
        if use_accelerator:
            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)

            decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered, ignore_pad_token_for_loss)
        else:
            decoded_preds, decoded_labels = postprocess(generated_tokens, labels, ignore_pad_token_for_loss)
        
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        #evalmetric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    results = metric.compute()
    #evalresults = evalmetric.compute()
    #print(f"BLEU score: {results['score']:.2f}")
    #print(evalresults['score'])
    return results

def evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    #evalprogress_bar = tqdm(range(num_training_steps))
    model.eval()
    all_start_logits = []
    all_end_logits = []
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        #Get the highest probability from the model output for the start and end positions:
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor: 384
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len) #(5043, 384)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len) #(5043, 384)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    predictions = (start_logits_concat, end_logits_concat)
    #prediction = post_processing_function(raw_datasets[valkey], eval_dataset, outputs_numpy)
    eval_examples = raw_datasets[valkey]
    # Post-processing: we match the start logits and end logits to answers in the original context.
    version_2_with_negative = False
    max_answer_length = 30
    n_best_size=20
    null_score_diff_threshold = 0.0
    stage="eval"
    predictions = predictions
    predictions = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=trainoutput,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    prediction = EvalPrediction(predictions=formatted_predictions, label_ids=references)
    #result=metric.compute(predicted_answers, theoretical_answers)
    #metric = evaluate.load("squad_v2" if version_2_with_negative else "squad")
    #result = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    result = metric.compute(prediction.predictions, prediction.label_ids)

    #start_logits = np.concatenate(all_start_logits) #8, 384 array to (102,384)
    #end_logits = np.concatenate(all_end_logits)
    # dataset_len=len(validation_dataset) #103
    # start_logits = start_logits[: dataset_len]
    # end_logits = end_logits[: dataset_len] #no size change
    # predicted_answers, theoretical_answers = QApostprocessing(
    #     start_logits, end_logits, validation_dataset, raw_datasets[valkey]
    # )#predicted_answers list of dict['id','prediction_text']; list of dict['id','answers['text', 'answer_start']']
    # result=metric.compute(predicted_answers, theoretical_answers)
    #print(f"epoch {epoch}, evaluation result:", result)
    return result



import shutil
def savemodels(model, optimizer, epoch, trainoutput):
    modelfilepath=os.path.join(trainoutput, 'savedmodel.pth')
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, modelfilepath)
    modelfilepathwithepoch=os.path.join(trainoutput, 'epoch'+str(epoch)+'_savedmodel.pth')
    shutil.copy(modelfilepath, modelfilepathwithepoch)
    #Huggingface format:
    model.save_pretrained(trainoutput)

#data_name="imdb", dataconfig="", model_checkpoint="distilbert-base-uncased"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="squad",
                    help='data name: opus_books, kde4, opus100, cnn_dailymail, billsum, xsum')
    parser.add_argument('--dataconfig', type=str, default='',
                    help='train_asks[:5000]')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache",
                    help='path to get data ') #r"E:\Dataset\NLPdataset\aclImdb"
    parser.add_argument('--model_checkpoint', type=str, default="mybert",
                    help='Model checkpoint name from HF, distilbert-base-uncased, t5-small, t5-base, Helsinki-NLP/opus-mt-en-zh, Helsinki-NLP/opus-mt-en-fr, t5-small, facebook/wmt21-dense-24-wide-en-x')
    parser.add_argument('--task', type=str, default="QA",
                    help='NLP tasks: translation, summarization, QA')
    parser.add_argument('--hfevaluate', default=True, action='store_true',
                    help='perform evaluation via HFevaluate or localevaluate')
    parser.add_argument('--dualevaluate', default=False, action='store_true',
                    help='perform evaluation via HFevaluate and localevaluate')
    parser.add_argument("--source_lang", type=str, default="en", help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default="fr", help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="1124",
                    help='Name the current training')
    parser.add_argument('--training', default=True, action='store_true',
                    help='Perform training')
    parser.add_argument('--usehpc', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', default=False, action='store_true',
                    help='Use Huggingface accelerator')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=16, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help=(
            "Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=384, #128, #1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    args = parser.parse_args()

    print("useHFevaluator:", args.hfevaluate)
    print("dualevaluator:", args.dualevaluate)

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    use_accelerator = args.useHFaccelerator
    model_checkpoint = args.model_checkpoint
    use_fp16 = True

    if args.source_prefix is None and args.model_checkpoint in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        print("You're running a t5 model but didn't provide a source prefix, which is the expected, e.g., translate English to Chinese: or summarize: ")
    
    USE_HPC=args.usehpc
    if USE_HPC:
        #https://huggingface.co/docs/transformers/installation#offline-mode
        #HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
        os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        os.environ['HF_EVALUATE_OFFLINE'] = "1"
        os.environ['HF_DATASETS_OFFLINE'] = "1"
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        os.environ['https_proxy'] = "https://172.16.1.2:3128"
        os.environ['HTTPS_PROXY'] = "https://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        #taskname=args.traintag #"eli5asksciencemodeling"
    else:
        trainoutput=args.outputdir #"./output"
        #taskname=args.traintag #taskname="eli5asksciencemodeling"
        mycache_dir="./data/"
    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    model, tokenizer, starting_epoch = loadmodel(model_checkpoint, task=task, mycache_dir=mycache_dir, pretrained=args.pretrained, hpc=USE_HPC, unfreezename=args.unfreezename)
    #tokenizer.model_max_len=512
    print(tokenizer.pad_token)
    print(tokenizer.eos_token)
    #tokenizer.pad_token = tokenizer.eos_token
    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if args.source_lang is not None:
            tokenizer.src_lang = args.source_lang
        if args.target_lang is not None:
            tokenizer.tgt_lang = args.target_lang

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            args.target_lang is not None and args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)
    #conflict with QA
    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    raw_datasets, text_column, target_column, task_column = loaddata(args, USE_HPC)
    column_names = raw_datasets["train"].column_names
    print("column_names:", column_names) #['translation']
    padding = "max_length" if args.pad_to_max_length else False
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    ignore_pad_token_for_loss = True

    if task in ["translation", "summarization"]:
        def seqpreprocess_function(examples):
            if task == "translation":
                inputs = [ex[text_column] for ex in examples[task_column]] # "translation"
                targets = [ex[target_column] for ex in examples[task_column]]
            elif task == "summarization":
                inputs = examples[text_column]
                targets = examples[target_column]
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized_datasets = raw_datasets.map(
                seqpreprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=raw_datasets["train"].column_names,
            )#The default batch size is 1000, but you can adjust it with the batch_size argument
        tokenized_datasets.set_format("torch")
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets[valkey]
    elif task in ['qa', 'QA', 'QuestionAnswering']:
        def QApreprocess_function(examples, mode='train'):
            questions = [ex.strip() for ex in examples[task_column]] #"question"
            context = examples[text_column] #"context"
            stride = 128
            model_inputs = tokenizer(
                questions,
                context, #examples["context"],
                max_length=args.max_source_length, #384
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True, #map the start and end positions of the answer to the original context 
                padding=padding, #"max_length",
            )
            if mode=='train':
                #add "start_positions" and "end_positions" into the inputs as the labels
                model_inputs=updateQAtraininputs(model_inputs, examples, tokenizer)
            else: #val
                #add "example_id"
                model_inputs=updateQAvalinputs(model_inputs, examples)
            return model_inputs
        
        # tokenized_datasets = raw_datasets.map(
        #         QApreprocess_function,
        #         batched=True,
        #         num_proc=1,
        #         remove_columns=raw_datasets["train"].column_names,
        #     )#The default batch size is 1000, but you can adjust it with the batch_size argument
        #tokenized_datasets = {}#raw_datasets.copy()
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets[valkey]
        mode='train'
        train_dataset = train_dataset.map(
            QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names,
                fn_kwargs={"mode": mode})
        #['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
        #inputs["overflow_to_sample_mapping"]=[0, 0, 0, 0] means one example split into 4 parts (features)
        mode='val'
        #small_eval_set = raw_datasets[valkey].select(range(500))
        eval_dataset =eval_dataset.map(
            QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names,
                fn_kwargs={"mode": mode}) 
        #tokenized_datasets[valkey] = validation_dataset.remove_columns(["example_id", "offset_mapping"]) 
        #eval_set_for_model = validation_dataset.remove_columns(["example_id", "offset_mapping"])
        #print(validation_dataset.features.keys())#['input_ids', 'attention_mask', 'offset_mapping', 'example_id']
        #print(eval_set_for_model.features.keys())#['input_ids', 'attention_mask']
        #tokenized_datasets[valkey] = validation_dataset.remove_columns(["offset_mapping"]) 
        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        print(eval_dataset.features.keys())#['input_ids', 'attention_mask', 'offset_mapping', 'example_id']
        print(eval_dataset_for_model.features.keys())#['input_ids', 'attention_mask']

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator #DefaultDataCollator()
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if use_fp16 else None,
        )

    #To test this on a few samples
    batch = data_collator([train_dataset[i] for i in range(1, 3)])
    print(batch.keys()) #['input_ids', 'attention_mask', 'labels'], dict_keys(['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    #batch["labels"] #our labels have been padded to the maximum length of the batch, using -100:
    #batch["decoder_input_ids"] #shifted versions of the labels

    metric = myEvaluator(args, useHFevaluator=args.hfevaluate, dualevaluator=args.dualevaluate)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size
    )

    #optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = get_myoptimizer(model, learning_rate=args.learningrate)

    num_train_epochs = args.total_epochs
    #num_update_steps_per_epoch = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    completed_steps = starting_epoch * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if use_accelerator:
        #accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        device = accelerator.device
        print("Using HF Accelerator and device:", device)
    else:
        accelerator = None
        if torch.cuda.is_available():
            device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model.to(device)
        print("Using device:", device)
    
    
    if task in ["translation", "summarization"]:
        evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, args.num_beams, metric)
    elif task in ['qa', 'QA', 'QuestionAnswering']:
        #evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric)
        evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput)

    if args.training == True:
        print("Start training, total steps:", num_training_steps)
        progress_bar = tqdm(range(num_training_steps))
        model.train()
        for epoch in range(starting_epoch, num_train_epochs):
            # Training
            for step, batch in enumerate(train_dataloader):
                #batch = {k: v.to(device) for k, v in batch.items()}
                if not use_accelerator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                if not use_accelerator:
                    loss.backward()
                else:
                    accelerator.backward(loss)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

            # Evaluation
            #results = evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, args.num_beams, metric)
            if task in ["translation", "summarization"]:
                results = evaluate_dataset(model, tokenizer, eval_dataloader, use_accelerator, accelerator, device, max_target_length, args.num_beams, metric)
            elif task in ['qa', 'QA', 'QuestionAnswering']:
                results = evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput)

            #print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
            print(f"epoch {epoch}, evaluation metric: {metric.metricname}")
            print("Evaluation result:", results)
            #print(evalresults['score'])
            # Save the results
            with open(os.path.join(trainoutput, f"epoch{epoch}_"+"eval_results.json"), "w") as f:
                #json.dump({"eval_bleu": results["score"]}, f)
                json.dump(results, f)

            if use_accelerator:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                #unwrapped_model.save_pretrained(trainoutput, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(trainoutput)
                savemodels(model, optimizer, epoch, trainoutput)
            else:
                #model.save_pretrained(trainoutput)
                #torch.save(model.state_dict(), os.path.join(trainoutput, 'savedmodel.pth'))
                savemodels(model, optimizer, epoch, trainoutput)
                tokenizer.save_pretrained(trainoutput)

    del model, optimizer, lr_scheduler
    if use_accelerator:
        accelerator.free_memory()

