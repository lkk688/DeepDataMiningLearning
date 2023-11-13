from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
import torch
import os

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            [" ".join(x) for x in examples["answers.text"]],
            padding="max_length",
            truncation=True,
        )

def tokenize_function(examples):
    result = tokenizer(examples["text"]) #generate "input_ids" list and "attention_mask"
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))] #one to one map to word_ids
    return result

def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    #print(examples.keys())
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    #print('total_length:', total_length)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def testpredictmask(model, text):
    #text = "This is a great [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits #torch.Size([1, 8, 30522])
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :] #torch.Size([1, 30522])
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    return top_5_tokens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="imdb",
                    help='data name: eli5, imdb')
    parser.add_argument('--dataconfig', type=str, default="",
                    help='train_asks[:5000]')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default=r"E:\Dataset\NLPdataset\aclImdb",
                    help='path to get data')
    parser.add_argument('--model_checkpoint', type=str, default="distilbert-base-uncased",
                    help='Model checkpoint name from https://huggingface.co/models, "bert-base-cased", "distilbert-base-uncased" "cardiffnlp/twitter-roberta-base-emotion"')
    parser.add_argument('--task', type=str, default="token_classifier",
                    help='NLP tasks: sentiment, token_classifier, "sequence_classifier", custom_classifier')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="eli5asksciencemodeling",
                    help='Name the current training')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--usehpc', type=bool, default=True,
                    help='Use HPC')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    model_checkpoint = args.model_checkpoint
    
    USE_HPC=args.usehpc
    if USE_HPC:
        mycache_dir="/data/cmpe249-fa23/Huggingfacecache"
        os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        #os.environ['https_proxy'] = "https://172.16.1.2:3128"
        #os.environ['HTTPS_PROXY'] = "https://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        taskname=args.traintag #"eli5asksciencemodeling"
    else:
        trainoutput="./output"
        taskname=args.traintag #taskname="eli5asksciencemodeling"

    if args.data_type == "huggingface":
        datasetpath="/data/cmpe249-fa23/Huggingfacecache/imdb/plain_text/1.0.0/d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0"
        if not args.dataconfig:
            #raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir) #eli5
            trainarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-train.arrow')
            #valarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-validation.arrow')
            testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test.arrow')
            raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath, 'test': testarrowpath})
        else:
            raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
        #Download to home/.cache/huggingface/dataset
        
        print(raw_datasets["train"][0])#multiple key value "text", "label"
        print("features:", raw_datasets["train"].features)
        #extract the text subfield from its nested structure with the flatten method:
        raw_datasets = raw_datasets.flatten() #nested structure become flat: answers.text
        #features: ['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers.a_id', 'answers.text', 'answers.score', 'title_urls.url', 'selftext_urls.url', 'answers_urls.url']

        print("All keys in raw datasets:", raw_datasets.keys())
    
    global tokenizer
    if USE_HPC:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(localpath)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)#, cache_dir=mycache_dir)
    tokenizer.model_max_len=512

    if USE_HPC:
        localpath=os.path.join(mycache_dir, model_checkpoint) #modelname="distilroberta-base"
        model = AutoModelForMaskedLM.from_pretrained(localpath)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)#"distilroberta-base")
    
    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> DistilBERT number of parameters: {round(model_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")

    text = "This is a great [MASK]."
    top_5_tokens = testpredictmask(model, text)

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

    # tokenizer(raw_datasets["train"][0]['answers.text'])
    # tokenizer_wrapper = TokenizerWrapper(tokenizer)
    # colnames = raw_datasets["train"].column_names
    # tokenized_dataset = raw_datasets.map(tokenizer_wrapper.tokenize_function, batched=True, num_proc=3, remove_columns=colnames)
    # print(tokenized_dataset)
    # lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )
    # Slicing produces a list of lists for each feature
    tokenized_samples = tokenized_datasets["train"][:3] #generate "input_ids" list and "attention_mask", add "word_ids"

    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length: {len(sample)}'") #length of the input_ids
    
    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    print(f"'>>> Concatenated reviews length: {total_length}'")
    print(tokenized_samples.keys())

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    tokenizer.pad_token = tokenizer.eos_token #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #mlm_probability means the percentage of [MASK]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) 
    samples = [lm_datasets["train"][i] for i in range(2)]
    # for sample in samples:
    #     _ = sample.pop("word_ids")
    # for chunk in data_collator(samples)["input_ids"]:
    #     print(f"\n'>>> {tokenizer.decode(chunk)}'")
    # for chunk in data_collator(samples)["input_ids"]:
    #     print(f"\n'>>> {tokenizer.convert_ids_to_tokens(chunk)}'")

    downsampled_dataset = raw_datasets["train"].train_test_split(test_size=0.2)


    
    
    training_args = TrainingArguments(
        output_dir=os.path.join(trainoutput, model_checkpoint, taskname), #"./output/my_awesome_eli5_mlm_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=True,
    )