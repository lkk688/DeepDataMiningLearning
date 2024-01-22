import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, features, load_metric
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
from transformers import Wav2Vec2Processor
from dataclasses import dataclass

from hfutil import valkey, TrustRemoteCode

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

#data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
@dataclass
class MyDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest" #= True
    task: Optional[str] = None
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length, #not used
            pad_to_multiple_of=self.pad_to_multiple_of, #not used
            return_tensors="pt",
        )

        if self.task and self.task == "audio-asr": #CTC padding
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            #with self.processor.as_target_processor():
            # labels_batch = self.processor.pad(
            #     labels = label_features,
            #     padding=self.padding,
            #     max_length=self.max_length_labels,
            #     pad_to_multiple_of=self.pad_to_multiple_of_labels,
            #     return_tensors="pt",
            # )
            # with self.processor.as_target_processor():
            #     labels_batch = self.processor.pad(
            #         label_features,
            #         padding=self.padding,
            #         return_tensors="pt",
            #     )
            labels_batch = self.processor.pad(labels=label_features, 
                                              padding=self.padding,max_length=self.max_length_labels,
                                              pad_to_multiple_of=self.pad_to_multiple_of_labels,
                                              return_tensors="pt",)
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        else:
            label_features = [feature["labels"] for feature in features]
            d_type = torch.long if isinstance(label_features[0], int) else torch.float
            batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

def loaddata(args, USE_HPC, mycache_dir):
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
            elif args.data_name == "speech_commands": #AUDIO
                raw_datasets = DatasetDict()
                raw_datasets["train"] = load_dataset(args.data_name, args.dataconfig, split='train')
                #raw_datasets[valkey] = load_dataset(args.data_name, args.dataconfig, split='validation')
                #raw_datasets = load_dataset(args.data_name, args.dataconfig, split='test')
                task_column ="audio" #(['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id']
                text_column = "audio"
                target_column = "label"
            elif args.data_name == "marsyas/gtzan":
                raw_datasets = load_dataset(args.data_name, "all")
                task_column ="audio" 
                text_column = "audio"
                target_column = "genre"
            elif args.data_name =="common_language":
                raw_datasets = load_dataset(args.data_name)
                task_column ="audio" 
                text_column = "audio"
                target_column = "language" #['client_id', 'path', 'audio', 'sentence', 'age', 'gender', 'language']
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
            elif args.data_name in ['squad', 'squad_v2']: #QA
                raw_datasets = load_dataset(args.data_name)
                #raw_datasets = load_dataset("squad", split="train[:5000]") #'train', 'test'
                #raw_datasets["train"][0] #'id', 'title','context', 'question', 'answers' (dict with 'text' and 'answer_start'),  
                task_column ="question"
                text_column = "context"
                target_column = "answers"
            #AUDIO Classification part
            elif args.data_name == "speech_commands": 
                raw_datasets = DatasetDict()
                raw_datasets["train"] = load_dataset(args.data_name, args.dataconfig, split='train', cache_dir=mycache_dir)
                #raw_datasets[valkey] = load_dataset(args.data_name, args.dataconfig, split='validation')
                #raw_datasets = load_dataset(args.data_name, args.dataconfig, split='test')
                task_column ="audio" #(['file', 'audio', 'label', 'is_unknown', 'speaker_id', 'utterance_id']
                text_column = "audio"
                target_column = "label"
            elif args.data_name == "marsyas/gtzan":
                raw_datasets = load_dataset(args.data_name, "all", cache_dir=mycache_dir)
                task_column ="audio" 
                text_column = "audio"
                target_column = "genre"
            elif args.data_name =="common_language":
                raw_datasets = load_dataset(args.data_name, cache_dir=mycache_dir)
                task_column ="audio" 
                text_column = "audio"
                target_column = "language" #['client_id', 'path', 'audio', 'sentence', 'age', 'gender', 'language']
            elif args.data_name =="common_voice": #not available
                Train_SAMPLES = 10000
                data_split=f"train[:{Train_SAMPLES}]" #"train+validation"
                raw_datasets = load_dataset("mozilla-foundation/common_voice_11_0", args.dataconfig, split=data_split, cache_dir=mycache_dir, trust_remote_code=TrustRemoteCode) #"en" "zh-CN"
                task_column ="audio" 
                text_column = "audio"
                target_column = "sentence"
            elif args.data_name.endswith("minds14"):
                #https://huggingface.co/datasets/PolyAI/minds14 contains recordings of people asking an e-banking system questions in several languages and dialects, and has the intent_class for each recording
                if args.dataconfig:
                    subsetconfig = args.dataconfig
                else:
                    subsetconfig = "all" #"en-AU" "zh-CN"
                raw_datasets = load_dataset("PolyAI/minds14", name=subsetconfig, split="train", cache_dir=mycache_dir)
                #raw_datasets = raw_datasets.train_test_split(test_size=0.2)
                #minds can be used to classify intent_class, lang_id, and speech recognition (english_transcription)
                #contains "path", "audio"dict("path", "array")
                task_column ="audio" 
                text_column = "path"
                if args.task=="audio-classification":
                    if args.subtask.startswith("intent"):
                        target_column = "intent_class"
                    else:
                        target_column = "lang_id"
                else:
                    target_column = "english_transcription"
            elif args.data_name == "superb":
                #https://huggingface.co/datasets/superb#ks
                if args.dataconfig:
                    subsetconfig = args.dataconfig
                else:
                    subsetconfig = "ks" #Keyword Spotting (KS)
                raw_datasets = load_dataset("superb", name=subsetconfig, split="train", ignore_verifications=True, cache_dir=mycache_dir)
                task_column ="audio" 
                text_column = "file"
                target_column = "label"
            elif args.data_name == "google/fleurs":
                raw_datasets = load_dataset("google/fleurs", "all", split="train", cache_dir=mycache_dir)
                task_column ="audio" 
                text_column = "path"
                target_column = "lang_id" #language
            #Audio asr dataset
            elif args.data_name == "timit":
                data_dir=os.path.join(mycache_dir, "timit", "TIMIT")
                raw_datasets = load_dataset("timit_asr", data_dir=data_dir)
                task_column ="audio" 
                text_column = "file"
                target_column = "text"
            elif args.data_name == "librispeech_asr":
                raw_datasets = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=mycache_dir)
                task_column ="audio" 
                text_column = "file"
                target_column = "text" 
            else: 
                #raw_datasets = load_dataset(args.data_name, args.dataconfig) #dataconfig="train_asks[:5000]"
                raw_datasets = load_dataset(args.data_name)
                text_column = "text"
                target_column = "summary"
        
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
        elif args.task.startswith("audio"):
            oneexample = split_datasets["train"][1]
            print("Audio: ", oneexample[task_column])
            print("Audio target: ", oneexample[target_column])
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

    elif args.data_type == "custom":
        #/DATA10T/Cache/aesdd/data
        datafolder=os.path.join(args.data_path, args.data_name, 'data')
        save_path=os.path.join(args.data_path, args.data_name)
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
def getlabels_asr(raw_datasets, task_column="audio", target_column="text", vocabpath=None): 
    show_random_elements(raw_datasets["train"].remove_columns(task_column))

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
