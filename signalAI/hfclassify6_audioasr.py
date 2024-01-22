#2023/12/4 modified based on huggingfaceSequence5, remove NLP related
#ref： https://github.com/huggingface/transformers/blob/main/examples/pytorch/audio-classification/run_audio_classification.py
#https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, features, load_metric
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, Wav2Vec2CTCTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, DataCollatorWithPadding, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator, EvalPrediction)
from transformers import (
    AutoProcessor,
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    EvalPrediction,
    set_seed,
)
from sklearn.metrics import classification_report
from transformers import TrainingArguments
import evaluate
import torch
import os
import librosa
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
import torchaudio
import math
import collections
import numpy as np
import random
import json
from random import randint
valkey='test'
TrustRemoteCode = True
import datetime
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
#torch version>1.6

#https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/modeling_outputs.py
from transformers.modeling_outputs import TokenClassifierOutput
#Optin1: TokenClassifierOutput, Option2: MyClassifierOutput
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
@dataclass
class MyModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

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

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

class MyClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)#768
        self.dropout = nn.Dropout(0.5) #config.final_dropout
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    #def forward(self, x):
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

#ref: https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2022
#Wav2Vec2ForCTC
class MyWave2Vec2ClassificationCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, basemodel_name, task="audio-classification", mycache_dir=None, pooling_mode='mean'):
        super().__init__(config)

        self.pooling_mode = pooling_mode#['mean', 'sum', 'max']
        if basemodel_name.find("Wav2Vec2") and basemodel_name:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(basemodel_name,cache_dir = mycache_dir, trust_remote_code=TrustRemoteCode) #("facebook/wav2vec2-base-960h")
            # config = AutoConfig.from_pretrained(
            #     pretrained_model_name_or_path=basemodel_name,
            # )
            # config.num_labels = num_classes #num_labels
            # self.config = AutoConfig.from_pretrained(
            #     basemodel_name,
            #     num_labels=num_classes, #len(labels),
            #     label2id=label2id,
            #     id2label=id2label,
            #     finetuning_task=task, #"audio-classification",
            #     cache_dir=mycache_dir,
            # )
            self.config = config
        elif config:
            self.wav2vec2 = Wav2Vec2Model(config)
            self.config = config
        else:
            print("Error in MyWave2Vec2Classification init!")
        
        print(self.wav2vec2.feature_extractor) #Wav2Vec2FeatureEncoder

        if task=="audio-classification":
            #self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            #num_classes = len(id2label)
            self.classifier = MyClassificationHead(config=self.config)
            self.num_labels = self.config.num_labels
            self.init_weights()
        elif task=="audio-asr":
            self.dropout = nn.Dropout(config.final_dropout)
            #self.target_lang = target_lang
            print("Config vocab size:", config.vocab_size)
            output_hidden_size = (
                config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
            )#1024
            self.lm_head_new = nn.Linear(output_hidden_size, config.vocab_size)
            # Initialize weights and apply final processing
            #self.post_init()
        


    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()
    
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #not used
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values, #[1, 57216]
            attention_mask=attention_mask, #[1, 57216]
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )#Wav2Vec2BaseModelOutput last_hidden_state:torch.Size([1, 312, 768])
        hidden_states = outputs[0] #[1, 178, 768]
        if task == "audio-classification":
            if self.pooling_mode == 'mean':
                x = torch.mean(hidden_states, dim=1) #torch.Size([1, 768])
            elif self.pooling_mode == 'sum':
                x = torch.sum(hidden_states, dim=1)
            else: #'max'
                x = torch.max(hidden_states, dim=1)[0]
            logits = self.classifier(x)#(hidden_states) torch.Size([1, 45])
        elif task == "audio-asr":
            hidden_states = self.dropout(hidden_states)
            logits = self.lm_head_new(hidden_states)

        loss = None
        if labels is not None and task == "audio-classification":
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss() #[64, 45], 64 is batchsize, 45 is number of labels' 
                # loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        elif labels is not None and task=="audio-asr":
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
                the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.
            """
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            _HIDDEN_STATES_START_POSITION = 2
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return MyModelOutput( #TokenClassifierOutput
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#https://github.com/huggingface/transformers/tree/main/src/transformers/models/wav2vec2
def loadmodel(model_checkpoint, custommodel=False, task="audio-classification", id2label=None, label2id=None, vocab_path=None, pretrained="", cache_dir="", unfreezename="", freeze_feature_encoder=True, freeze_base_model=True, return_attention_mask=True):
    ignore_mismatched_sizes = custommodel #when loading model, ignore size 
    if cache_dir:
        mycache_dir = cache_dir
    elif os.environ.get('HF_HOME') is not None:
        mycache_dir = os.environ.get('HF_HOME')

    #option1
    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #Option2
    #newtokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    #config.vocab_size:32
    unk_token="[UNK]"
    pad_token="[PAD]"
    word_delimiter_token="|"

    #Create Processor
    processoroption1=False
    if processoroption1==True:
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint, cache_dir=mycache_dir,return_attention_mask=return_attention_mask)
        feature_extractor = processor.feature_extractor
    else:
        config = AutoConfig.from_pretrained(model_checkpoint)#config.model_type:'wav2vec2' config.tokenizer_class=None
        tokenizer_type = config.model_type if config.tokenizer_class is None else None #'wav2vec2'
        #config = config if config.tokenizer_class is not None else None #config.tokenizer_class = None
        tokenizer_kwargs = {
            "config": config,
            "tokenizer_type": tokenizer_type,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }
        if vocab_path and task =="audio-asr":
            # tokenizer = AutoTokenizer.from_pretrained(
            #     vocab_path, #vocab_filepath,
            #     #**tokenizer_kwargs,
            #     cache_dir = mycache_dir,
            #     #config=config,#None
            #     # tokenizer_type=tokenizer_type, #'wav2vec2'
            #     do_lower_case=True,
            #     # unk_token=unk_token,
            #     # pad_token=pad_token,
            #     word_delimiter_token=word_delimiter_token,
            #     )
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                vocab_path, #vocab_filepath,
                cache_dir = mycache_dir,
                unk_token=unk_token,
                pad_token=pad_token,
                word_delimiter_token=word_delimiter_token,
                trust_remote_code=TrustRemoteCode,
                )
        elif task =="audio-asr":
            #tokenizer_name_or_path=model_checkpoint #"anton-l/wav2vec2-tokenizer-turkish" #model_checkpoint
            # tokenizer = AutoTokenizer.from_pretrained(
            #     tokenizer_name_or_path,
            #     #token=data_args.token,
            #     #trust_remote_code=data_args.trust_remote_code,
            #     **tokenizer_kwargs,
            # )
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, \
                                                    cache_dir = mycache_dir,\
                                                    do_lower_case=True, \
                                                    word_delimiter_token="|",
                                                    trust_remote_code=TrustRemoteCode)
        #test tokenizer
        print("Tokenizer encoder len:", len(tokenizer.encoder))
        print("Tokenizer len:", len(tokenizer))
        #outputids = tokenizer("test the tokenizer")
        #tokenizer.save_pretrained("./output")
        #if datatype=="huggingface":

        if model_checkpoint:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint, cache_dir=mycache_dir, do_normalize=True,return_attention_mask=return_attention_mask, trust_remote_code=TrustRemoteCode)
        else:
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        #processor = None
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    if task == "audio-classification":
        useconfig=False #choose different options
        if custommodel: #
            #option1:
            #configuration = Wav2Vec2Config()
            #option2:
            configuration = AutoConfig.from_pretrained(
                model_checkpoint,
                num_labels=len(id2label),
                label2id=label2id,
                id2label=id2label,
                finetuning_task=task, #"audio-classification",
                cache_dir=mycache_dir,
            )
            model = MyWave2Vec2ClassificationCTC(config=configuration,basemodel_name=model_checkpoint, task=task, mycache_dir=mycache_dir)
        elif useconfig==True:
            # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
            # transformer outputs in the classifier, but it doesn't always lead to better accuracy
            # model_args.feature_extractor_name or model_args.model_name_or_path,
            # return_attention_mask=model_args.attention_mask,
            # cache_dir=model_args.cache_dir,
            #Option1
            config = AutoConfig.from_pretrained(
                model_checkpoint,
                num_labels=len(label2id), #len(labels),
                label2id=label2id,
                id2label=id2label,
                finetuning_task=task, #"audio-classification",
                cache_dir=mycache_dir,
            )
            model = AutoModelForAudioClassification.from_pretrained(
                model_checkpoint,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                cache_dir=mycache_dir,
            )
        else:
            #Option2
            model = AutoModelForAudioClassification.from_pretrained(
                model_checkpoint, 
                num_labels=len(label2id),
                label2id=label2id,
                id2label=id2label,
                cache_dir=mycache_dir,
            )
            #ignore_mismatched_sizes: Will enable to load a pretrained model whose head dimensions are different.
    elif task == "audio-asr":
        if custommodel: #
            #option1:
            #configuration = Wav2Vec2Config()
            #option2:
            configuration = AutoConfig.from_pretrained(
                model_checkpoint,
                cache_dir=mycache_dir,
                trust_remote_code=True,
            )
            configuration.update(
                {
                    "final_dropout": 0.0,
                    "mask_time_prob": 0.05,
                    #"mask_time_length": model_args.mask_time_length,
                    #"mask_feature_prob": model_args.mask_feature_prob,
                    #"mask_feature_length": model_args.mask_feature_length,
                    "gradient_checkpointing": True,
                    "layerdrop": 0.0, #The LayerDrop probability
                    "ctc_loss_reduction": "mean",
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "vocab_size": len(processor.tokenizer),
                    "adapter_attn_dim": 16,
                }
            )
            model = MyWave2Vec2ClassificationCTC(config=configuration,basemodel_name=model_checkpoint, task=task, mycache_dir=mycache_dir)

            # pretrained_model = AutoModelForCTC.from_pretrained(model_checkpoint) 
            # model.load_state_dict(pretrained_model.state_dict(), strict= False)
        else:
            config = AutoConfig.from_pretrained(model_checkpoint)
            print("Config vocab size", config.vocab_size)
            print(len(processor.tokenizer))
            #processor.tokenizer.add_tokens("_")
            #print(len(processor.tokenizer))
            model = AutoModelForCTC.from_pretrained(
                model_checkpoint, 
                #ignore_mismatched_sizes=True,
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer), #processor.tokenizer.vocab_size,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                feat_proj_dropout=0.0,
                mask_time_prob=0.05,
                layerdrop=0.1,
            )#The tokenizer's pad_token_id must be to define the model's pad_token_id or in the case of a CTC speech model also CTC's blank token
            # model = Wav2Vec2ForCTC.from_pretrained(
            #     model_checkpoint, #"facebook/wav2vec2-xls-r-300m",
            #     cache_dir = mycache_dir,
            #     # attention_dropout=0.0,
            #     # hidden_dropout=0.0,
            #     # feat_proj_dropout=0.0,
            #     # mask_time_prob=0.05,
            #     # layerdrop=0.0,
            #     ctc_loss_reduction="mean",
            #     pad_token_id=processor.tokenizer.pad_token_id,
            #     vocab_size=len(processor.tokenizer),
            # )

    starting_epoch = 0
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        starting_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['model_state_dict'])

    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")

    # freeze the convolutional waveform encoder
    if hasattr(model, "freeze_feature_extractor"):#not used
        model.freeze_feature_extractor()
    if freeze_feature_encoder:
        #model.freeze_feature_extractor()
        model.freeze_feature_encoder()
    if freeze_base_model:
        model.freeze_base_model()
    modelparameters(model, unfreezename)
    return model, feature_extractor, processor, starting_epoch

# max_length = 128
# def preprocess_function(examples):
#     inputs = [ex[source_lang] for ex in examples["translation"]] #1000
#     targets = [ex[target_lang] for ex in examples["translation"]] #1000
#     model_inputs = globaltokenizer(
#         inputs, text_target=targets, max_length=max_length, truncation=True
#     )
#     return model_inputs

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


def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    if weight_decay>0:
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
    else:
        adam_beta1=0.9
        adam_beta2=0.999
        adam_epsilon=1e-8
        optimizer = AdamW(
            list(model.parameters()),
            lr=learning_rate,
            betas=[adam_beta1, adam_beta2],
            eps=adam_epsilon,
        )
    return optimizer

def loaddefaultmodel_fromname(modelname="facebook/wav2vec2-base-960h", task="audio-asr", cache_dir="./output"):
    if task == "audio-classification":
        feature_extractor = AutoFeatureExtractor.from_pretrained(modelname, cache_dir=cache_dir)
        model = AutoModelForAudioClassification.from_pretrained(modelname, cache_dir=cache_dir)
        tokenizer = None
        processor = None
    elif task == "audio-asr":
        model = AutoModelForCTC.from_pretrained(modelname, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir=cache_dir)
        feature_extractor = AutoFeatureExtractor.from_pretrained(modelname, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(modelname, cache_dir=cache_dir)
    return model, feature_extractor, tokenizer, processor

def settarget_lang(model, processor, target_lang='eng'):
    processor.tokenizer.set_target_lang(target_lang) #"cmn-script_simplified"
    model.load_adapter(target_lang)

import librosa
def asrinference_path(datasample, model, samplerate=16_000, processor=None, device='cuda'):
    if isinstance(datasample, str):#batch["path"] get numpyarray
        datasamples, sampling_rate = librosa.load(datasample, sr=samplerate)
        batchdecode=False
    elif isinstance(datasample, list):
        datasamples=[]
        for sample in datasample:
            onespeech, sampling_rate = librosa.load(sample, sr=samplerate)
            datasamples.append(onespeech)
        batchdecode=True
    
    inputs = processor(datasamples, sampling_rate=samplerate, return_tensors="pt", padding=True)
    print("Input:", type(inputs))

    model=model.to(device)
    inputs=inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    print("Output logits shape:", outputs.shape)
    if batchdecode:
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    else:
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
    return transcription

from transformers import pipeline
def inferencesample(datasample, task, model, usepipeline=True, feature_extractor=None, processor=None, device='cuda', task_column='audio', target_column='text'):
    sample = datasample[task_column] #raw_datasets["train"][0][task_column]
    print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

    saveaudio_tofile(sample['array'], "./output", sample["sampling_rate"], filename="save_example.wav")
    #loadaudio_fromfile(filepath=sample["path"], plotfig=True)

    if usepipeline:
        mypipeline = pipeline(
            task, #"audio-classification",
            model=model, #"anton-l/xtreme_s_xlsr_300m_minds14",
        )
        #sample = datasample[columnname]
        result = mypipeline(sample)#sample can be audio (sample["audio"]) or path (minds[0]["path"])
        print(result)
    else:
        #sample=datasample[columnname]
        #sample["audio"]["array"]
        print(feature_extractor.sampling_rate)
        print(sample['sampling_rate']) #dataset.features["audio"].sampling_rate 
        
        if isinstance(model, str):
            model, feature_extractor, tokenizer, processor = loaddefaultmodel_fromname(modelname=model, task=task)
        
        inputs = feature_extractor(sample["array"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")#size did not change
        # inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        print(f"inputs keys: {list(inputs.keys())}")#[1, 49562]
        input_values=inputs['input_values'].numpy()#(1, 48640)
        print(
            f"Mean: {np.mean(input_values):.3}, Variance: {np.var(input_values):.3}"
        )
        if processor:
            inputs2 = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
            input_values=inputs2['input_values'].numpy()
            print(
                f"Mean: {np.mean(input_values):.3}, Variance: {np.var(input_values):.3}"
            )
            #test tokenizer
            #tokenids = processor.tokenizer(datasample[target_column]).input_ids
            #decoded_str = processor.tokenizer.batch_decode(tokenids, group_tokens=False)

        model=model.to(device)
        inputs=inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        #Get the class with the highest probability
        predicted_ids=torch.argmax(logits, dim=-1) #[1, 45]->33; [1, 113, 32]->[1, 113]

        if task == "audio-classification":
            predicted_class_ids = predicted_ids.item() #.item() moves the scalar data to CPU
            #use the model’s id2label mapping to convert it to a label:
            result = model.config.id2label[str(predicted_class_ids)]
            print(result)
            #list all classes
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            allclasses = [{"class": model.config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
        elif task == "audio-asr":
            # transcribe speech
            transcription = processor.batch_decode(predicted_ids)
            result = transcription[0]
            print(transcription[0])
            # retrieve word stamps (analogous commands for `output_char_offsets`)
            outputs = processor.tokenizer.decode(np.squeeze(predicted_ids), output_word_offsets=True)
            print(outputs["text"])
            text=datasample[target_column]
            # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate 
            time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
            word_offsets = [
                {
                    "word": d["word"],
                    "start_time": round(d["start_offset"] * time_offset, 2),
                    "end_time": round(d["end_offset"] * time_offset, 2),
                }
                for d in outputs.word_offsets
            ]
            print(word_offsets[:3])

    return result
    
def inferenceaudiopath(data_path, task, model, usepipeline=True, feature_extractor=None, device='cuda', columnname='audio'):
    if isinstance(data_path, str):
        speech_array, _sampling_rate = torchaudio.load(data_path)
        resampler = torchaudio.transforms.Resample(feature_extractor.sampling_rate)
        audio_np = resampler(speech_array).squeeze().numpy()
    #elif isinstance(data_path, np.array):
    else:
        audio_np = data_path
    
    features = feature_extractor(audio_np, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    
    logitsmax=torch.argmax(logits) #[1, 45]->33
    predicted_class_ids = logitsmax.item() #.item() moves the scalar data to CPU
    #use the model’s id2label mapping to convert it to a label:
    result = model.config.id2label[str(predicted_class_ids)]
    print(result)
    
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Target": model.config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    print(f"file path: {data_path} \t predict: {result} \t score:{scores[predicted_class_ids]} ")

    return result, outputs






class myEvaluator:
    def __init__(self, args, useHFevaluator=False, dualevaluator=False, labels=None, processor=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = args.task
        self.preds = []
        self.refs = []
        self.labels = labels
        self.processor = processor
        self.HFmetric = None
        if args.task == "audio-classification":
            self.metricname = "accuracy" #"mse" "wer"
        elif args.task == "audio-asr":
            self.metricname = "wer"
        self.LOmetric = None
        #https://huggingface.co/spaces/evaluate-metric/wer
        #if self.task.startswith("audio"):
        if self.useHFevaluator:
            # Load the accuracy metric from the datasets package
            self.HFmetric = evaluate.load(self.metricname) #evaluate.load("mse")
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
            self.preds.clear()
            self.refs.clear()
        return results
    
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

#data_name="marsyas/gtzan", dataconfig="", model_checkpoint="ntu-spml/distilhubert"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #data related arguments
    parser.add_argument('--data_type', type=str, default="huggingface",
                    help='data type name: huggingface, custom')
    parser.add_argument('--data_name', type=str, default="common_voice",
                    help='data name: common_voice, librispeech_asr, aesdd(local path), timit, common_language, superb, google/fleurs, minds14, marsyas/gtzan')
    parser.add_argument('--dataconfig', type=str, default='zh-CN',
                    help='dataset_config_name, e.g., common_voice subset en, zh-CN')
    parser.add_argument('--subset', type=float, default=0,
                    help='0 means all dataset')
    parser.add_argument('--data_path', type=str, default="/DATA10T/Cache", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    #model related arguments
    parser.add_argument('--model_checkpoint', type=str, default="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                    help='Model checkpoint name from HF, jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn, facebook/mms-1b-all, jonatasgrosman/wav2vec2-large-xlsr-53-english, TencentGameMate/chinese-wav2vec2-base, facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-large-xlsr-53, anton-l/xtreme_s_xlsr_300m_minds14, facebook/wav2vec2-base-960h, "facebook/wav2vec2-base", ntu-spml/distilhubert')
    parser.add_argument('--checkpointfolder', type=str, default="",
                    help='Model training checkpoint to resume')
    parser.add_argument('--pretrained', type=str, default="",
                    help='Pretrained model path')
    parser.add_argument('--custommodel', default=False, action='store_true', help='Change model') 
    parser.add_argument('--task', type=str, default="audio-asr",
                    help='tasks: audio-classification, openqa, translation, summarization, QA')
    parser.add_argument('--subtask', type=str, default="intent-classification",
                    help='Sub tasks')
    parser.add_argument('--hfevaluate', default=True, action='store_true',
                    help='perform evaluation via HFevaluate or localevaluate')
    parser.add_argument('--dualevaluate', default=False, action='store_true',
                    help='perform evaluation via HFevaluate and localevaluate')
    parser.add_argument('--unfreezename', type=str, default="",
                    help='Unfreezename in models')
    parser.add_argument('--freeze_feature_encoder', default=False, action='store_true', help='Freeze the featureencoder')
    parser.add_argument('--freeze_basemodel', default=False, action='store_true', help='Freeze the basemodel')
    #training related arguments
    parser.add_argument('--outputdir', type=str, default="/DATA10T/output/", help='output path') #r"E:\output" "./output" "/DATA10T/output/"
    parser.add_argument('--traintag', type=str, default="0122",
                    help='Name the current training')
    # parser.add_argument('--training', default=True, action='store_true',
    #                 help='Perform training')
    parser.add_argument('--trainmode', default="CustomTrain", choices=['HFTrainer','CustomTrain', 'NoTrain'], help='Training mode')
    #vocab_path
    parser.add_argument('--use_vocabpath', default=False, action='store_true', help='Use HPC')
    parser.add_argument('--use_fp16', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--use_gradientcheckpoint', default=True, action='store_true',
                    help='Use gradientcheckpoint')#gradient_checkpointing_enable
    parser.add_argument('--usehpc', default=False, action='store_true',
                    help='Use HPC')
    parser.add_argument('--useHFaccelerator', default=False, action='store_true',
                    help='Use Huggingface accelerator')
    parser.add_argument('--useamp', default=True, action='store_true',
                    help='Use pytorch amp in training')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learningrate', default=3e-4, type=float, help='Learning rate')
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    #warmup_ratio
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps in learning rate scheduling.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass, ref: https://kozodoi.me/blog/20210219/gradient-accumulation.",
    )
    parser.add_argument(
        "--max_length_seconds",
        type=int,
        default=8, #20,
        help=(
            "Audio clips will be randomly cut to this length during training if the value is set.."
        ),
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

    USE_HPC=args.usehpc
    if USE_HPC:
        #https://huggingface.co/docs/transformers/installation#offline-mode
        #HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        mycache_dir=args.data_path #"/data/cmpe249-fa23/Huggingfacecache"
        #os.environ['TRANSFORMERS_CACHE'] = mycache_dir
        os.environ['HF_HOME'] = mycache_dir
        os.environ['HF_DATASETS_CACHE'] = mycache_dir
        #os.environ['HF_EVALUATE_OFFLINE'] = "1"
        #os.environ['HF_DATASETS_OFFLINE'] = "1"
        #os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['http_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTP_PROXY'] = "http://172.16.1.2:3128"
        os.environ['https_proxy'] = "http://172.16.1.2:3128"
        os.environ['HTTPS_PROXY'] = "http://172.16.1.2:3128"
        trainoutput="/data/cmpe249-fa23/trainoutput/huggingface"
        #taskname=args.traintag #"eli5asksciencemodeling"
    else:
        if os.path.exists(args.data_path):
            mycache_dir=args.data_path
            os.environ['HF_HOME'] = mycache_dir
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        elif os.environ.get('HF_HOME') is not None:
            mycache_dir=os.environ['HF_HOME']
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        else:
            mycache_dir="./data/"
            os.environ['HF_HOME'] = mycache_dir
            os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # mycache_dir=os.path.join('D:',os.sep, 'Cache','huggingface')
        
        print("HF_HOME:", os.environ['HF_HOME'])
        print("HF_DATASETS_CACHE:", os.environ['HF_DATASETS_CACHE'])
        # os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # if os.environ.get('HF_HOME') is None:
        #     mycache_dir=args.data_path
        #     os.environ['HF_HOME'] = mycache_dir
        #     os.environ['HF_DATASETS_CACHE'] = mycache_dir
        # else:
        #     print("HF_HOME:", os.environ['HF_HOME'])
        #     mycache_dir=os.environ['HF_HOME']
        trainoutput=args.outputdir #"./output"
        #taskname=args.traintag #taskname="eli5asksciencemodeling"
    
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(args.gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        args.useamp = False
    else:
        device = torch.device("cpu")
        args.useamp = False

    trainoutput=os.path.join(trainoutput, model_checkpoint, args.data_name+'_'+args.traintag)
    os.makedirs(trainoutput, exist_ok=True)
    print("Trainoutput folder:", trainoutput)

    raw_datasets, text_column, target_column, task_column, column_names= loaddata(args, USE_HPC, mycache_dir)
    #labels, id2label, label2id, column_names, columns_remove = getlabels(raw_datasets, task_column, target_column)
    labels = None
    id2label = None
    label2id = None
    tokenizer=None
    vocab_path = None
    if args.task == "audio-classification":
        labels, id2label, label2id, num_labels = getlabels_classifier(raw_datasets, target_column=target_column, datatype=args.data_type)
    elif args.task == "audio-asr":
        if args.use_vocabpath:
            if args.dataconfig:
                vocab_path = os.path.join(mycache_dir, args.data_name, args.dataconfig)
            else:
                vocab_path = os.path.join(mycache_dir, args.data_name)
            #vocab_path = "./signalAI/"
        raw_datasets = getlabels_asr(raw_datasets, task_column=task_column, target_column=target_column, vocabpath=vocab_path)

    model, feature_extractor, processor, starting_epoch = \
        loadmodel(model_checkpoint, custommodel=args.custommodel, \
                task=task, id2label=id2label, label2id=label2id, \
                vocab_path=vocab_path, pretrained=args.pretrained, \
                unfreezename=args.unfreezename, freeze_feature_encoder=args.freeze_feature_encoder, freeze_base_model=args.freeze_basemodel, return_attention_mask=True)

    model_input_name = feature_extractor.model_input_names[0]
    print("model_input_name:", model_input_name) #input_values
    model = model.to(device)
    if args.use_gradientcheckpoint:
        #https://huggingface.co/docs/transformers/main_classes/model
        model.gradient_checkpointing_enable() #save GPU memory 
        #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) #https://pytorch.org/docs/stable/checkpoint.html

    
    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        task_column, features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    #inference example
    rand_idx = randint(0, len(raw_datasets["train"])-1)
    datasample=raw_datasets["train"][rand_idx]
    result=inferencesample(datasample=datasample, task = args.task, model=model, usepipeline=False, feature_extractor=feature_extractor, processor=processor, device=device, task_column=task_column, target_column=target_column)


    def prepare_asrdataset(batch):
        audio = batch[task_column] #"audio"

        # batched output is "un-batched" to ensure mapping is correct
        #https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
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
                    speech, max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            audio_list.append(speech_sub)
        for label in examples[target_column]:
            #add label
            #label = example #example[target_column]
            if len(labels) > 0:
                target = labels.index(label) if label in labels else -1
                target_list.append(target)
        
        inputs = feature_extractor(audio_list, 
                                   sampling_rate=feature_extractor.sampling_rate,max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), truncation=True, padding=True)
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
            max_length=int(feature_extractor.sampling_rate * args.max_length_seconds),
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
                    audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
                subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs, 
                                       sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
                                       truncation=True,
                                       padding=True)
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[target_column])
        else:
            audio = batch[task_column]
            wav = random_subsample(
                    audio["array"], max_length=args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
                )
            subsampled_wavs.append(wav)
            inputs = feature_extractor(subsampled_wavs,sampling_rate=feature_extractor.sampling_rate,
                                       max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), 
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
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate, max_length=int(feature_extractor.sampling_rate * args.max_length_seconds), truncation=True, padding=True)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[target_column])

        return output_batch


    if args.data_type == "huggingface" and args.task=="audio-classification":
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
    elif args.data_type == "custom" and args.task=="audio-classification":
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
    elif args.data_type == "huggingface" and args.task=="audio-asr":
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
        
    # Load the accuracy metric from the datasets package
    #metric = evaluate.load("accuracy")
    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    # def compute_metrics(eval_pred):
    #     """Computes accuracy on a batch of predictions"""
    #     predictions = np.argmax(eval_pred.predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    metriceval = myEvaluator(args, useHFevaluator=args.hfevaluate, dualevaluator=args.dualevaluate, labels=labels, processor=processor)
    #metriceval.compute_metrics
    #result1=metriceval.compute(predictions=[2,7,6,1], references=[2,7,7,7])
    #result2=metriceval.compute(predictions=[2,7,6,1,1], references=[2,7,6,1,1])

    if processor:
        #processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        data_collator = MyDataCollatorWithPadding(processor=processor, padding=True, task=args.task)
        #the padding tokens in the labels with -100 so that those tokens are not taken into account when computing the loss.
    else:
        data_collator = default_data_collator
    
    #['HFTrainer','CustomTrain', 'NoTrain']
    if args.trainmode == 'HFTrainer':
        training_args = TrainingArguments(
            trainoutput,
            group_by_length=True,#makes training more efficient by grouping training samples of similar input length into one batch.
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learningrate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.total_epochs,
            #warmup_ratio=args.warmup_ratio, #0.1,
            warmup_steps=args.warmup_steps, #500,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            #metric_for_best_model=myEvaluator.#"accuracy",
            fp16=args.use_fp16,
            push_to_hub=False,
            #gradient_checkpointing=True,#reduce GPU memory, or use model.gradient_checkpointing_enable()
        )
        # Initialize our trainer
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets[valkey],
            compute_metrics=metriceval.compute_metrics,
            tokenizer=processor.feature_extractor, #feature_extractor,
        )

        if args.checkpointfolder and os.path.exists(args.checkpointfolder):
            checkpoint = args.checkpointfolder #"output/facebook/wav2vec2-base/minds14_0105/checkpoint-14704"
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        #Perform evaluation
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    elif args.trainmode == 'CustomTrain':
        
        train_dataloader = DataLoader(
            raw_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            raw_datasets[valkey], collate_fn=data_collator, batch_size=args.batch_size
        )

        results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)

        # Optimizer
        optimizer = get_myoptimizer(model, learning_rate=args.learningrate, weight_decay=args.weight_decay)
        
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = (args.total_epochs - starting_epoch) * num_update_steps_per_epoch

        num_warmup_steps = args.warmup_steps #10
        lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type, #["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max_train_steps,
            )
        
        #recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps /  num_update_steps_per_epoch)
        total_batch_size = args.batch_size * args.gradient_accumulation_steps

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0
        #starting_epoch = 0
        if args.useamp == True:
            # Creates a GradScaler once at the beginning of training.
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(starting_epoch, num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                # Enables autocasting for the forward pass (model + loss)
                if args.useamp == True:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**batch)
                        loss = outputs.loss
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                #backward
                if args.useamp == True:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if args.useamp == True:
                            scaler.step(optimizer)
                            # Updates the scale for next iteration.
                            scaler.update()
                        else:
                            optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1
            #Evaluation
            results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)   
            print(f"epoch {epoch}, loss: {loss}, evaluation: {metriceval.metricname}")
            print("Evaluation result:", results)
            # Save the results
            with open(os.path.join(trainoutput, f"epoch{epoch}_"+"eval_results.json"), "w") as f:
                #json.dump({"eval_bleu": results["score"]}, f)
                json.dump(results, f)
            
            savemodels(model, optimizer, epoch, trainoutput)
            #feature_extractor.save_pretrained(trainoutput)
    else: #NoTrain
        eval_dataloader = DataLoader(
            raw_datasets[valkey], collate_fn=None, batch_size=args.batch_size
        )
        results=evaluate_dataset(model, eval_dataloader, device, metriceval, processor=processor)
        print("No training, evauate only:", results)

    # using now() to get current time
    current_time = datetime.datetime.now()
    # Printing value of now.
    print("Time now is:", current_time)
    print("Finished")


    

