from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import TrainingArguments
import torch
import os
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import math
import collections
import numpy as np

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1,2" #"0,1"
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"
#print(os.environ['CUDA_VISIBLE_DEVICES'])
# num_gpus= torch.cuda.device_count()
# print("Device numbers:", num_gpus)
# for gpuidx in range(num_gpus):
#     print("Device properties:", torch.cuda.get_device_properties(gpuidx))
#     print("Utilization:", torch.cuda.utilization(gpuidx))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(gpuidx)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(gpuidx)/1024**3,1), 'GB')

global data_field
data_field='answers.text'
global block_size
block_size = 512 #128

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_function(self, examples):
        result =  self.tokenizer(
            [" ".join(x) for x in examples[data_field]], #"answers.text"
            padding="max_length",
            truncation=True,
        )
        if self.tokenizer.is_fast: #always fast
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))] #one to one map to word_ids
            #each token is mapped to the word it originates from, if some tokens belongs to the same word, it will show the duplicate the wordids
        return result

def tokenize_function(examples):
    result = tokenizer(examples[data_field]) #"text" generate "input_ids" list and "attention_mask"
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))] #one to one map to word_ids
        #each token is mapped to the word it originates from, if some tokens belongs to the same word, it will show the duplicate the wordids
    return result

#split the concatenated sequences into shorter chunks defined by block_size, which should be both shorter than the maximum input length and short enough for your GPU RAM.
def group_texts(examples):
    #block_size = 128
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
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    wwm_probability = 0.2
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)

#insert MASK into the original dataset for evaluator
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = evaldata_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def testpredictmask(model, text, device='cuda'):
    #text = "This is a great [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    inputs=inputs.to(device)
    model=model.to(device)
    print(inputs)
    token_logits = model(**inputs).logits #torch.Size([1, 8, 30522]) shape (batch_size, sequence_length, hidden_size)
    # Find the location of [MASK] and extract its logits
    findmask=torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    mask_token_index = findmask[1] #[1]
    if (mask_token_index.numel()):
        mask_token_logits = token_logits[0, mask_token_index, :] #torch.Size([1, 30522])
        # Pick the [MASK] candidates with the highest logits
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    else:
        print("Did not find the mask token.")
    return top_5_tokens

def testgenerate(model, text, device='cuda'):
    inputs = tokenizer(text, return_tensors="pt").input_ids
    inputs=inputs.to(device)
    model=model.to(device)
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_text

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def loadmodel(model_checkpoint, task="CLM", mycache_dir="", pretrained="", hpc=True):
    if hpc==True:
        localpath=os.path.join(mycache_dir, model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(localpath)
        if task=="CLM":
            model = AutoModelForCausalLM.from_pretrained(localpath)
        else:
            model = AutoModelForMaskedLM.from_pretrained(localpath)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)#, cache_dir=mycache_dir)
        if task=="CLM":
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)#"distilroberta-base")
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer

#data_name="imdb", dataconfig="", model_checkpoint="distilbert-base-uncased"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="eli5",
                    help='data name: eli5, imdb')
    parser.add_argument('--dataconfig', type=str, default='',
                    help='train_asks[:5000]')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/data/cmpe249-fa23/Huggingfacecache",
                    help='path to get data ') #r"E:\Dataset\NLPdataset\aclImdb"
    parser.add_argument('--model_checkpoint', type=str, default="distilroberta-base",
                    help='Model checkpoint name from h ttps://huggingface.co/models, distilgpt2 "distilroberta-base", "bert-base-cased", "distilbert-base-uncased" "cardiffnlp/twitter-roberta-base-emotion"')
    parser.add_argument('--task', type=str, default="MLM",
                    help='NLP tasks: MLM, CLM, LLM')
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    parser.add_argument('--outputdir', type=str, default="./output",
                    help='output path')
    parser.add_argument('--traintag', type=str, default="1116MLM",
                    help='Name the current training')
    parser.add_argument('--training', type=bool, default=True,
                    help='Perform training')
    parser.add_argument('--usehpc', type=bool, default=False,
                    help='Use HPC')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=8, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=2e-5, type=float, help='Learning rate')
    args = parser.parse_args()

    global task
    task = args.task
    print(' '.join(f'{k}={v}' for k, v in vars(args).items())) #get the arguments as a dict by calling vars(args)

    use_accelerator = False
    if task == "MLM":
        CausalLM=False
        mlm_probability = 0.15
        WHOLE_Word = True
        InsertMask = True
    else:
        CausalLM=True
        mlm = False
        WHOLE_Word = False
        InsertMask = False

    model_checkpoint = args.model_checkpoint
    
    USE_HPC=args.usehpc
    if USE_HPC:
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
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
        trainoutput=args.outputdir #"./output"
        taskname=args.traintag #taskname="eli5asksciencemodeling"
    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)
    
    if args.data_type == "huggingface":
        if USE_HPC:
            if args.data_name=='imdb':
                datasetpath=os.path.join(mycache_dir, args.data_name, "plain_text", "1.0.0", "d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0") #"/data/cmpe249-fa23/Huggingfacecache/imdb/plain_text/1.0.0/d613c88cf8fa3bab83b4ded3713f1f74830d1100e171db75bbddb80b3345c9c0"
                #raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir) #eli5
                trainarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-train.arrow')
                #valarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-validation.arrow')
                testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath, 'test': testarrowpath})
                data_field='text'
                #sampletext = "This is a great [MASK]."
            elif args.data_name=='eli5':
                #datasetpath="/data/cmpe249-fa23/Huggingfacecache/eli5/LFQA_reddit/1.0.0/17574e5502a10f41bbd17beba83e22475b499fa62caa1384a3d093fc856fe6fa"
                datasetpath=os.path.join(mycache_dir, args.data_name, "LFQA_reddit", "1.0.0", "17574e5502a10f41bbd17beba83e22475b499fa62caa1384a3d093fc856fe6fa")
                trainarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-train'+'_asks.arrow')#eli5-train_asks.arrow
                testarrowpath=os.path.join(mycache_dir, datasetpath, args.data_name+'-test'+'_asks.arrow')
                raw_datasets = load_dataset("arrow", data_files={'train': trainarrowpath, 'test': testarrowpath})
                data_field='answers.text'
                #sampletext = "A static force applied eccentric to the center of [MASK]."#mass
                if CausalLM:
                    sampletext = "Somatic hypermutation allows the immune system to"
                else: #MLM
                    sampletext = "The Milky Way is a <mask> galaxy."
                raw_datasets = raw_datasets.flatten() #nested structure become flat: answers.text
                #features: ['q_id', 'title', 'selftext', 'document', 'subreddit', 'answers.a_id', 'answers.text', 'answers.score', 'title_urls.url', 'selftext_urls.url', 'answers_urls.url']
        else:
            if args.data_name=='eli5':
                train_ds, test_ds = load_dataset(args.data_name, split=['train_asks', 'test_asks']) #dataconfig="train_asks[:5000]"
                raw_datasets= DatasetDict({
                    'train': train_ds,
                    'test': test_ds
                })
                data_field='answers.text'
                raw_datasets = raw_datasets.flatten() #nested structure become flat: answers.text
                if CausalLM:
                    sampletext = "Somatic hypermutation allows the immune system to"
                else: #MLM
                    sampletext = "The Milky Way is a <mask> galaxy."
            else:
                raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
        #Download to home/.cache/huggingface/dataset
        
        print(raw_datasets["train"][0])#multiple key value "text", "label"
        print("features:", raw_datasets["train"].features)
        #extract the text subfield from its nested structure with the flatten method:
        
        print("All keys in raw datasets:", raw_datasets['train'][0].keys())
        if args.subset>0:
            if args.subset<1:
                trainlen=int(len(raw_datasets["train"])*args.subset)
                testlen=int(len(raw_datasets["test"])*args.subset)
            else:
                trainlen = int(min(args.subset, len(raw_datasets["train"])))
                testlen = int(trainlen/10)
            print("trainlen:", trainlen)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets["test"] = raw_datasets["test"].shuffle(seed=42).select([i for i in list(range(testlen))])
        

    global tokenizer
    model, tokenizer = loadmodel(model_checkpoint, task=task, mycache_dir=mycache_dir, pretrained=args.pretrained, hpc=USE_HPC)
    tokenizer.model_max_len=512
    tokenizer_wrapper = TokenizerWrapper(tokenizer)
    if CausalLM:
        tokenizer.pad_token = tokenizer.eos_token

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")
    #print(f"'>>> BERT number of parameters: 110M'")
    modelparameters(model, args.unfreezename)

    examples=raw_datasets["train"][0]
    listexamples = [" ".join(x) for x in examples[data_field]]
    token_train=tokenizer(listexamples)
    
    #device='cpu' #'cuda:1'
    device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
    if CausalLM:
        result = testgenerate(model, sampletext, device)
        print(result)
    else:
        top_5_tokens = testpredictmask(model, sampletext, device)
        for token in top_5_tokens:
            print(f"'>>> {sampletext.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

    # tokenizer(raw_datasets["train"][0]['answers.text'])
    # tokenizer_wrapper = TokenizerWrapper(tokenizer)
    # colnames = raw_datasets["train"].column_names
    # tokenized_dataset = raw_datasets.map(tokenizer_wrapper.tokenize_function, batched=True, num_proc=3, remove_columns=colnames)
    # print(tokenized_dataset)
    # lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

    # Use batched=True to activate fast multithreading!
    # tokenized_datasets = raw_datasets.map(
    #     tokenize_function, batched=True, remove_columns=["text", "label"]
    # )
    print(data_field)#='answers.text'
    tokenized_datasets = raw_datasets.map(tokenizer_wrapper.tokenize_function, batched=True, num_proc=1, remove_columns=raw_datasets["train"].column_names)
    # Slicing produces a list of lists for each feature
    tokenized_samples = tokenized_datasets["train"][:3] #generate "input_ids" list and "attention_mask", add "word_ids"

    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Text {idx} length: {len(sample)}'") #length of the input_ids
    
    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    print(f"'>>> Concatenated reviews length: {total_length}'")
    print(tokenized_samples.keys())

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=8)

    #tokenizer.pad_token = tokenizer.eos_token #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(tokenizer.eos_token) #None </s>
    print(tokenizer.pad_token) #[PAD]

    #mlm_probability means the percentage of [MASK]
    global evaldata_collator
    if CausalLM:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        evaldata_collator = data_collator #DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        if WHOLE_Word:
            data_collator = whole_word_masking_data_collator
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
            samples = [lm_datasets["train"][i] for i in range(2)]
            for sample in samples:
                _ = sample.pop("word_ids")
            for chunk in data_collator(samples)["input_ids"]:
                print(f"\n'>>> {tokenizer.decode(chunk)}'")
            for chunk in data_collator(samples)["input_ids"]:
                print(f"\n'>>> {tokenizer.convert_ids_to_tokens(chunk)}'") 
        
        evaldata_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
        

    
    downsampled_dataset = lm_datasets["train"].train_test_split(test_size=0.2)

    batch_size = args.batch_size #64
    if CausalLM:
        downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
        test_dataset = downsampled_dataset["test"]
    else:
        test_dataset = downsampled_dataset["test"].remove_columns(["word_ids"])
    
    if InsertMask:
        eval_dataset = test_dataset.map(
            insert_random_mask,
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        eval_dataset = eval_dataset.rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )
    else:
        eval_dataset=test_dataset #downsampled_dataset["test"]
    
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    if use_accelerator:
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    num_train_epochs = args.total_epochs
    num_update_steps_per_epoch = len(train_dataloader) #2500
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    print("Total num_training_steps: ", num_training_steps)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    evalprogress_bar = tqdm(range(len(eval_dataloader)))
    if not use_accelerator:
        model=model.to(device)
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            if not use_accelerator:
                batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if not use_accelerator:
                loss.backward()
            else:
                accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        #progress_bar.refresh()  # force print final state
        #progress_bar.reset()
        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if not use_accelerator:
                    batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            loss = outputs.loss
            if not use_accelerator:
                losses.append(loss.repeat(batch_size))
            else:
                losses.append(accelerator.gather(loss.repeat(batch_size)))
            evalprogress_bar.update(1)
        evalprogress_bar.refresh()  # force print final state
        #evalprogress_bar.reset()

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        if use_accelerator:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(trainoutput, save_function=accelerator.save)
        else:
            #model.save_pretrained(trainoutput)
            #torch.save(model.state_dict(), os.path.join(trainoutput, 'savedmodel.pth'))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(trainoutput, 'savedmodel.pth'))
            # load
            # checkpoint = torch.load(output_model, map_location='cpu')
            # model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if accelerator.is_main_process:
        #     tokenizer.save_pretrained(output_dir)
        #     repo.push_to_hub(
        #         commit_message=f"Training in progress epoch {epoch}", blocking=False
        #     )
    progress_bar.close()
    evalprogress_bar.close()
    if CausalLM:
        result = testgenerate(model, sampletext, device)
        print(result)
    else:
        top_5_tokens = testpredictmask(model, sampletext, device)
        for token in top_5_tokens:
            print(f"'>>> {sampletext.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


    #text = "This is a great [MASK]"
    # inputs = tokenizer(sampletext, return_tensors="pt")
    # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    # #print(mask_token_index)
    # inputs=inputs.to('cuda')
    # logits = model(**inputs).logits
    # mask_token_logits = logits[0, mask_token_index, :]
    # #mask_token_logits
    # top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
    # for token in top_3_tokens:
    #     print(sampletext.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # training_args = TrainingArguments(
    #     output_dir=os.path.join(trainoutput, model_checkpoint, taskname), #"./output/my_awesome_eli5_mlm_model",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     num_train_epochs=10,
    #     weight_decay=0.01,
    #     push_to_hub=True,
    # )