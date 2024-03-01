#import matplotlib.pyplot as plt
import datasets
from datasets import load_dataset, DatasetDict, features, load_metric, Audio
import torch
import os
import librosa
from torch.utils.data import DataLoader
import torch
import torchaudio
import math
import collections
import numpy as np
import random
import json
from random import randint
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import functools
from DeepDataMiningLearning.hfaudio.hfutil import valkey, TrustRemoteCode

def savedict2file(data_dict, filename):
    # save vocab dict to be loaded into tokenizer
    with open(filename, "w") as file:
        json.dump(data_dict, file)

#https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#saving-audio-to-file
def saveaudio_tofile(audiowave, dir, sample_rate, filename="save_example.wav"):
    path=f"{dir}/{filename}"
    #Save without any encoding option. The function will pick up the encoding which the provided data fit
    #audiowave is 1D numpy 
    audiowave_tensor=torch.from_numpy(audiowave) #torch.Size([16000])
    audiowave_tensor=audiowave_tensor.unsqueeze(0)#1D to 2D tensor torch.Size([1, 16000])
    torchaudio.save(path, audiowave_tensor, sample_rate)
    #Save as 16-bit signed integer Linear PCM The resulting file occupies half the storage but loses precision
    #torchaudio.save(path, audiowave, sample_rate, encoding="PCM_S", bits_per_sample=16)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")

#https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data
def loadaudio_fromfile(filepath="save_example.wav", plotfig=True):
    #path=f"{dir}/{filename}"
    filesize = os.path.getsize(filepath)
    print(f" - File size: {filesize} bytes")
    print(f" - {torchaudio.info(filepath)}")
    if filesize>0: 
        #the resulting tensor object has dtype=torch.float32 and its value range is [-1.0, 1.0].
        waveform, sample_rate = torchaudio.load(filepath) #get tensor
        waveform_np = waveform.numpy()#[1, 16000]
        if plotfig==True:
            plot_waveformnp(waveform_np, sample_rate)

#each folder includes specific class with wav files, the folder name is the class name (similar format with image classification). We need to loop over directories and save the paths related to each class based on the directory name.
def loadaudios_todataset(folder, save_path, audiofileformat=".wav", target_columnname="target", valkey="test"):
    data = []
     #"emotion"
    for path in tqdm(Path(folder).glob("**/*"+audiofileformat)): #"**/*.wav"
        name = str(path).split('/')[-1].split('.')[0] #filename before .wav
        label = str(path).split('/')[-2] #folder name

        try:
            # There are some broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                target_columnname: label
            })
        except Exception as e:
            # print(str(path), e)
            pass

    df = pd.DataFrame(data)
    # Filter broken and non-existed paths
    print(f"Data len before filter: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop(labels="status", axis=1)
    print(f"Data len after filter: {len(df)}")
    labels = df[target_columnname].unique()
    print("Labels: ", labels)
    labels_count = df.groupby(target_columnname).count()[["path"]]
    print(labels_count)

    #get one sample
    idx = np.random.randint(0, len(df))
    sample = df.iloc[idx]
    path = sample["path"]
    label = sample[target_columnname]
    speech, sr = torchaudio.load(path) #torch.Size([1, 298614]), sr=44100
    speech = speech[0].numpy().squeeze() #numpy (298614,)
    #speech = librosa.resample(np.asarray(speech), sr, 16_000)
    speech = librosa.resample(y=np.asarray(speech), orig_sr=sr, target_sr=16_000)
    #import IPython.display as ipd
    #ipd.Audio(data=np.asarray(speech), autoplay=True, rate=16000)
    saveaudio_tofile(speech, "./output", 16_000, filename="fileaudio_example.wav")

    #split data into train test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df[target_columnname])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_path=f"{save_path}/train.csv"
    test_path=f"{save_path}/test.csv"
    train_df.to_csv(train_path, sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(test_path, sep="\t", encoding="utf-8", index=False)
    print(train_df.shape)
    print(test_df.shape)
    data_files = {
        "train": train_path,
        valkey: test_path,
    }
    return data_files


def plot_waveformnp(waveform, sample_rate):
    import matplotlib.pyplot as plt
    #waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

def load_audiodataset(data_name, dataconfig, mycache_dir, data_type="huggingface", task="audio-asr", subtask="", subset=0, data_path=None, USE_HPC=False):
    task_column =""
    text_column = ""
    target_column =""
    if data_type == "huggingface":
        #AUDIO Classification part
        if data_name == "speech_commands": 
            raw_datasets = DatasetDict()
            raw_datasets["train"] = load_dataset(data_name, dataconfig, split='train', cache_dir=mycache_dir)
            #raw_datasets[valkey] = load_dataset(args.data_name, args.dataconfig, split='validation')
            #raw_datasets = load_dataset(args.data_name, args.dataconfig, split='test')
            task_column ="audio" #(['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id']
            text_column = "audio"
            target_column = "label"
        elif data_name == "marsyas/gtzan":
            raw_datasets = load_dataset(data_name, "all", cache_dir=mycache_dir)
            task_column ="audio" 
            text_column = "audio"
            target_column = "genre"
        elif data_name =="common_language":
            raw_datasets = load_dataset(data_name, cache_dir=mycache_dir)
            task_column ="audio" 
            text_column = "audio"
            target_column = "language" #['client_id', 'path', 'audio', 'sentence', 'age', 'gender', 'language']
        elif data_name =="common_voice": #not available
            Train_SAMPLES = None #20000 #29000 #10000
            if Train_SAMPLES:
                data_split=f"train[:{Train_SAMPLES}]" #"train+validation"
            else:
                data_split="train"
            raw_datasets = load_dataset("mozilla-foundation/common_voice_11_0", dataconfig, split=data_split, cache_dir=mycache_dir, trust_remote_code=TrustRemoteCode) #"en" "zh-CN"
            task_column ="audio" 
            text_column = "audio"
            target_column = "sentence"
        elif data_name.endswith("minds14"):
            #https://huggingface.co/datasets/PolyAI/minds14 contains recordings of people asking an e-banking system questions in several languages and dialects, and has the intent_class for each recording
            if dataconfig:
                subsetconfig = dataconfig
            else:
                subsetconfig = "all" #"en-AU" "zh-CN"
            raw_datasets = load_dataset("PolyAI/minds14", name=subsetconfig, split="train", cache_dir=mycache_dir)
            #raw_datasets = raw_datasets.train_test_split(test_size=0.2)
            #minds can be used to classify intent_class, lang_id, and speech recognition (english_transcription)
            #contains "path", "audio"dict("path", "array")
            task_column ="audio" 
            text_column = "path"
            if task=="audio-classification":
                if subtask.startswith("intent"):
                    target_column = "intent_class"
                else:
                    target_column = "lang_id"
            else:
                target_column = "english_transcription"
        elif data_name == "superb":
            #https://huggingface.co/datasets/superb#ks
            if dataconfig:
                subsetconfig = dataconfig
            else:
                subsetconfig = "ks" #Keyword Spotting (KS)
            raw_datasets = load_dataset("superb", name=subsetconfig, split="train", ignore_verifications=True, cache_dir=mycache_dir)
            task_column ="audio" 
            text_column = "file"
            target_column = "label"
        elif data_name == "google/fleurs":
            raw_datasets = load_dataset("google/fleurs", "all", split="train", cache_dir=mycache_dir)
            task_column ="audio" 
            text_column = "path"
            target_column = "lang_id" #language
        #Audio asr dataset
        elif data_name == "timit":
            data_dir=os.path.join(mycache_dir, "timit", "TIMIT")
            raw_datasets = load_dataset("timit_asr", data_dir=data_dir)
            task_column ="audio" 
            text_column = "file"
            target_column = "text"
        elif data_name == "librispeech_asr":
            raw_datasets = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=mycache_dir)
            task_column ="audio" 
            text_column = "file"
            target_column = "text" 
        else: 
            #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
            raw_datasets = load_dataset(data_name)
            task_column ="audio" 
            text_column = "file"
            target_column = "text" 
        
        #Download to home/.cache/huggingface/dataset
        #print(raw_datasets.column_names)
        #splits=raw_datasets.split
        # print(raw_datasets.columns)
        if isinstance(raw_datasets.column_names, dict):
            print("All keys in raw datasets:", raw_datasets['train']) #obly one ['translation'] key
            split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
                # rename the "test" key to "validation" 
            #split_datasets["validation"] = split_datasets.pop("test")
        else: #no train/test split
            split_datasets = DatasetDict()
            split_datasets["train"] = raw_datasets
            split_datasets = split_datasets["train"].train_test_split(train_size=0.9, seed=20)
        
        #one element
        if task=="translation":
            sampletext = split_datasets["train"][1][task_column]
            print("Translation text: ", sampletext)
        elif task=="summarization":
            sampletext_text = split_datasets["train"][1][text_column]
            sampletext_target = split_datasets["train"][1][target_column]
            print("Summarization context: ", sampletext_text)
            print("Summarization target: ", sampletext_target)
        elif task=="QA":
            oneexample = split_datasets["train"][1]
            print("Context: ", oneexample[text_column])
            print("Question: ", oneexample[task_column])
            print("Answer: ", oneexample[target_column])#dict with 'text' and 'answer_start'
        elif task.startswith("audio"):
            oneexample = split_datasets["train"][1]
            print("Audio: ", oneexample[task_column])
            print("Audio target: ", oneexample[target_column])
        raw_datasets = split_datasets
        if subset>0:
            if subset<1:
                trainlen=int(len(raw_datasets["train"])*subset)
                testlen=int(len(raw_datasets[valkey])*subset)
            else:
                trainlen = int(min(subset, len(raw_datasets["train"])))
                testlen = int(trainlen/10)
            print("trainlen:", trainlen)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(trainlen))])
            raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(testlen))])

    elif data_type == "custom":
        #/DATA10T/Cache/aesdd/data
        datafolder=os.path.join(data_path, data_name, 'data')
        save_path=os.path.join(data_path, data_name)
        task_column = "path" #"audio" 
        text_column = "path"
        target_column = "label"
        data_files = loadaudios_todataset(folder=datafolder, save_path=save_path, audiofileformat=".wav", target_columnname=target_column, valkey=valkey)
        raw_datasets = load_dataset("csv", data_files = data_files, delimiter="\t")

    column_names = raw_datasets["train"].column_names
    print("column_names:", column_names)
    if task_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name not found in dataset. "
        )
    if target_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name not found in dataset. "
        )
    
    #remove other columns
    columnnames_remove = []
    for columnname in column_names:
        if columnname not in [task_column, target_column]:
            columnnames_remove.append(columnname)
    raw_datasets = raw_datasets.remove_columns(columnnames_remove)

    #limit the evaluation set size
    maxtestlen = 5000
    if len(raw_datasets[valkey])>maxtestlen:
        raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in list(range(maxtestlen))])
    #return column_names, labels, id2label, label2id
    return raw_datasets, text_column, target_column, task_column, column_names

def gettestdata(dataset_name, language='en', split="test", sampling_rate=16000, mycache_dir='./data', streaming=False, samples = 10, task_column="audio", target_column="sentence"):
    if samples and samples>0:
        splitwithsample=split+f"[:{samples}]"
    else:
        splitwithsample=split
    dataset_test = load_dataset(dataset_name, language, split=splitwithsample, cache_dir=mycache_dir, streaming=streaming) #streaming=True

    #remove other columns
    column_names = dataset_test.column_names
    columnnames_remove = []
    for columnname in column_names:
        if columnname not in [task_column, target_column]:
            columnnames_remove.append(columnname)
    dataset_test = dataset_test.remove_columns(columnnames_remove)
    #dataset_test = dataset_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=sampling_rate))

    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    def remove_special_characters(batch):
        batch[target_column] = re.sub(chars_to_ignore_regex, '', batch[target_column]).lower() + " "
        return batch
    dataset_test = dataset_test.map(remove_special_characters)
    
    return dataset_test
    #zh_sample = next(iter(dataset_test))["audio"]["array"]

def filter_datasetlength(datasets, min_duration_in_seconds, max_duration_in_seconds, sampling_rate=16_000, num_workers=1):
    max_input_length = max_duration_in_seconds * sampling_rate
    min_input_length = min_duration_in_seconds * sampling_rate

    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    # filter data that is shorter than min_input_length
    datasets = datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )
    return datasets

def dataset_castsamplingrate(raw_datasets, sampling_rate=16_000, audio_column_name="audio"):
    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[audio_column_name].sampling_rate
    if dataset_sampling_rate != sampling_rate:#feature_extractor.sampling_rate
        raw_datasets = raw_datasets.cast_column(
            audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
        )
    return raw_datasets

def getlabels_classifier(raw_datasets, target_column="label", datatype="huggingface"):
    if datatype=="huggingface":
        labels = raw_datasets["train"].features[target_column].names
        num_labels = len(labels)
        id2label_fn = raw_datasets["train"].features[target_column].int2str
        id2label = {
            str(i): id2label_fn(i)
            for i in range(len(labels))
        }
        label2id = {v: k for k, v in id2label.items()}
        #return labels, id2label, label2id
    else:
        labels = raw_datasets['train'].unique(target_column)
        labels.sort()  # Let's sort it for determinism
        num_labels = len(labels)
        print(f"A classification problem with {num_labels} classes: {labels}")
        label2id={label: i for i, label in enumerate(labels)}
        id2label={i: label for i, label in enumerate(labels)}
    return labels, id2label, label2id, num_labels

#from IPython.display import display, HTML
#show_random_elements(dataset["train"].remove_columns(["audio", "file"]), num_examples=10)
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    subset = dataset[picks]
    df = pd.DataFrame(subset)
    print(df)
    return subset
    #display(HTML(df.to_html()))

import re
def vocab_asr(raw_datasets, task_column="audio", target_column="text", vocabpath=None, a2z_only=False): 
    show_random_elements(raw_datasets["train"].remove_columns(task_column))

    def remove_characters(batch):#anything except these
        #batch[target_column] = re.sub(r'[a-z]+', '', batch[target_column])# remove latin characters
        batch[target_column] = re.sub('[^A-Za-z0-9 ]+', '', batch[target_column]) #re.sub(r'[^a-zA-Z]', '', mystring)
        #the a-zA-Z are character ranges that indicate all the lowercase and uppercase letter, respectively, and the caret ^ at the beginning of the character class indicates negation, e.g. "anything except these".
        return batch
    
    if a2z_only: #anyting except a-zA-Z
        raw_datasets = raw_datasets.map(remove_characters)

    #remove special characters, normalize the text to only have lower case letters and append a word separator token at the end.
    #chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
    def remove_special_characters(batch):
        replace_str = re.sub(chars_to_ignore_regex, '', batch[target_column])
        replace_str = replace_str.lower() + " "
        batch[target_column] =  replace_str
        return batch
    
    raw_datasets = raw_datasets.map(remove_special_characters)

    show_random_elements(raw_datasets["train"].remove_columns(task_column))

    if vocabpath: #use vocab
        vocab_filepath = os.path.join(vocabpath, 'vocab.json')
        #Create vocab
        if not (os.path.exists(vocab_filepath)):
            #extract all distinct letters
            def extract_all_chars(batch):
                all_text = " ".join(batch[target_column])
                vocab = list(set(all_text))
                return {"vocab": [vocab], "all_text": [all_text]}
            
            vocabs = raw_datasets.map(
                extract_all_chars,
                batched=True,
                batch_size=-1,
                keep_in_memory=True, 
                remove_columns=raw_datasets.column_names["train"]
                )
            #create the union of all distinct letters into an enumerated dictionary.
            vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs[valkey]["vocab"][0]))
            vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
            vocab_dict["|"] = vocab_dict[" "] #convert space to more visible character |
            del vocab_dict[" "]
            vocab_dict["[UNK]"] = len(vocab_dict)
            vocab_dict["[PAD]"] = len(vocab_dict)
            print("vocab dict:", vocab_dict)
            vocab_len=len(vocab_dict)
            print("vocab len:", vocab_len) #30

            os.makedirs(vocabpath, exist_ok=True)
            with open(vocab_filepath, 'w') as vocab_file:
                json.dump(vocab_dict, vocab_file)

    return raw_datasets

            
def dataset_removecharacters(raw_datasets, target_column="text", chars_to_ignore=None):    
    #remove special characters, normalize the text to only have lower case letters and append a word separator token at the end.
    if chars_to_ignore is not None:
        chars_to_ignore_regex = (
            f'[{"".join(chars_to_ignore)}]' 
        )
    else:
        #chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            replace_str = re.sub(chars_to_ignore_regex, '', batch[target_column])
            replace_str = replace_str.lower() + " "
            batch[target_column] =  replace_str
        else:
            batch[target_column] = batch[target_column].lower()
        return batch
    
    raw_datasets = raw_datasets.map(remove_special_characters)
    return raw_datasets

def getlabels(raw_datasets, task_column, target_column):
    labels = raw_datasets["train"].features[target_column].names
    column_names = raw_datasets["train"].column_names
    print("column_names:", column_names)
    if task_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name not found in dataset. "
        )
    if target_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name not found in dataset. "
        )
    id2label_fn = raw_datasets["train"].features[target_column].int2str
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(labels))
    }
    label2id = {v: k for k, v in id2label.items()}
    #print("label2id: ", label2id)
    #Option2
    # label2id, id2label = dict(), dict()
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label
    # newtarget_column='label'
    # raw_datasets["train"][newtarget_column] = raw_datasets["train"].pop(target_column) #split_datasets.pop("test")
    # raw_datasets[valkey][newtarget_column] = raw_datasets[valkey].pop(target_column)
    # newcolumn_names = raw_datasets["train"].column_names
    columns_remove = []
    for column in column_names:
        if not (column==target_column or column==task_column):
            columns_remove.append(column)
    return labels, id2label, label2id, column_names, columns_remove

def create_vocabulary_from_data(
    datasets: DatasetDict,
    dataset_key: Optional[str] = "train",
    target_column: Optional[str] = "text",
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch[target_column])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets[dataset_key].column_names,
    )

    # take union of all unique characters in each dataset
    # vocab_set = functools.reduce(
    #     lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]), vocabs.values()
    # )

    vocab_set = list(set(vocabs[dataset_key]["vocab"][0]))

    #vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs[valkey]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

def dataset_preprocessing(raw_datasets, tokenizer, processor, task_column="audio", target_column="text", max_length_seconds=10, model_input_name="input_values", labels=None, data_type = "huggingface", task="audio-asr", forward_attention_mask=False):
    feature_extractor = processor.feature_extractor
    #args.max_length_seconds
    #args.data_type == "huggingface" and args.task=="audio-asr"
    def prepare_asrdataset(batch):
        audio = batch[task_column] #"audio"

        # batched output is "un-batched" to ensure mapping is correct
        #https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"],return_attention_mask=forward_attention_mask)#.input_values[0]
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(batch[model_input_name]) #len(audio["array"])
        
        #forwards all its arguments to PreTrainedTokenizer
        # with processor.as_target_processor():
        #     batch["labels"] = processor(batch[target_column]).input_ids
        batch["labels"] = processor.tokenizer(batch[target_column]).input_ids
        return batch

    def preprocess_loadaudio(examples):
        audio_list=[]
        target_list=[]
        for path in examples[task_column]:
            #path=example #example[task_column] #'path'
            speech_array, sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(sampling_rate, feature_extractor.sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            speech_sub = random_subsample(
                    speech, max_length=max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            audio_list.append(speech_sub)
        for label in examples[target_column]:
            #add label
            #label = example #example[target_column]
            if len(labels) > 0:
                target = labels.index(label) if label in labels else -1
                target_list.append(target)
        
        inputs = feature_extractor(audio_list, 
                                   sampling_rate=feature_extractor.sampling_rate,max_length=int(feature_extractor.sampling_rate * max_length_seconds), truncation=True, padding=True)
        result = {model_input_name: inputs.get(model_input_name)}
        result["labels"] = list(target_list)
        #result["labels"] = example[target_column]
        return result

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples[task_column]] #"audio"
        target_labels = [np.int64(x) for x in examples[target_column]]
        #https://huggingface.co/docs/transformers/main_classes/feature_extractor
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_length_seconds),
            truncation=True,
            padding=True, #'max_length'
            return_attention_mask=True,
        )
        #return inputs
        output_batch = inputs #{model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = target_labels #list(target_labels) #list(examples[target_column])
        return output_batch

    # def preprocess_function_simple(examples):
    #     audio_arrays = [x["array"] for x in examples[task_column]] #examples[task_column] is list of audio data
    #     inputs = feature_extractor(
    #         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
    #         truncation=True,
    #         padding=True #'max_length'
    #     )
    #     return inputs
    
    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        subsampled_wavs = []
        #print(batch.keys())
        #print(batch[task_column])
        if isinstance(batch[task_column], list):
            for audio in batch[task_column]:
                wav = random_subsample(
                    audio["array"], max_length=max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
                subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs, 
                                       sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * max_length_seconds), 
                                       truncation=True,
                                       padding=True)
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[target_column])
        else:
            audio = batch[task_column]
            wav = random_subsample(
                    audio["array"], max_length=max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs,sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * max_length_seconds), 
                                       truncation=True,
            padding=True) #'max_length')
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = batch[target_column]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        if isinstance(batch[task_column], list):
            wavs = [audio["array"] for audio in batch[task_column]]
        else:
            audio = batch[task_column]
            wavs = [audio["array"]]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, max_length=int(feature_extractor.sampling_rate * max_length_seconds), truncation=True, padding=True)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[target_column])

        return output_batch


    if data_type == "huggingface" and task=="audio-classification":
        #samplebatch1 = preprocess_function_simple(raw_datasets['train'][:5])
        samplebatch2 = preprocess_function(raw_datasets['train'][:5])
        samplebatch3 = train_transforms(raw_datasets['train'][:5])
    
        processmode=1 #1 or 2 all works
        if processmode == 1:
            # Set the training transforms
            raw_datasets["train"].set_transform(train_transforms, output_all_columns=True)
            #transferred_datasets = DatasetDict()
            #transferred_datasets["train"]=raw_datasets["train"].map(train_transforms)
            # # Set the validation transforms
            raw_datasets[valkey].set_transform(val_transforms, output_all_columns=True)
            # #transferred_datasets[valkey]=raw_datasets[valkey].map(val_transforms)
            # train_dataset=raw_datasets["train"]
            # eval_dataset=raw_datasets[valkey]
            # #train_dataset=transferred_datasets["train"]
            # #eval_dataset=transferred_datasets[valkey]
        else:
            #columns_remove.append("audio")
            #print("columns_remove:", columns_remove)
            #https://huggingface.co/docs/datasets/about_map_batch
            raw_datasets = raw_datasets.map(
                train_transforms, #preprocess_function, preprocess_function_simple,
                remove_columns=column_names, #columns_remove,
                batched=True, #The primary objective of batch mapping is to speed up processing. The default batch size is 1000
                batch_size=100,
            )
        column_names = raw_datasets["train"].column_names
        # target_ready = False
        # input_ready = False
        # for column in column_names:
        #     if column == target_column:
        #         if target_column != "label":
        #             #change target_column name
        #             dataset_encoded = dataset_encoded.rename_column(target_column, "label")
        #             target_column="label"
        #             target_ready = True
        #         else:
        #             target_ready = True
        #     elif column == model_input_name:
        #         input_ready = True
        #     else:#remove column
        #         print("column name:", column)
        # column_names = raw_datasets["train"].column_names
        print(column_names)
    elif data_type == "custom" and task=="audio-classification":
        samplebatch1 = preprocess_loadaudio(raw_datasets['train'][:5])
        #custom dataset, audio is not loaded
        #preprocess_loadaudio
        raw_datasets = raw_datasets.map(
            preprocess_loadaudio,
            remove_columns=column_names, #columns_remove,
            batched=True, #The primary objective of batch mapping is to speed up processing. The default batch size is 1000
            batch_size=100,
        )
        column_names = raw_datasets["train"].column_names
    elif data_type == "huggingface" and task=="audio-asr":
        column_names = raw_datasets["train"].column_names
        samplebatch1 = prepare_asrdataset(raw_datasets['train'][1])
        settransformoption =False
        if settransformoption:
            raw_datasets["train"].set_transform(prepare_asrdataset, output_all_columns=True)
            raw_datasets[valkey].set_transform(prepare_asrdataset, output_all_columns=True)
        else:
            raw_datasets = raw_datasets.map(
                prepare_asrdataset,
                remove_columns=column_names, #columns_remove,
                #batched=True, #The primary objective of batch mapping is to speed up processing. The default batch size is 1000
                #batch_size=100,
            )
    
    return raw_datasets