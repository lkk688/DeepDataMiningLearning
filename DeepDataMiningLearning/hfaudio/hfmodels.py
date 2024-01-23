from transformers import (AutoConfig, AutoTokenizer, pipeline, AutoProcessor,
                          AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCTC,
                            Wav2Vec2Model,
                            Wav2Vec2Config,
                            Wav2Vec2ForCTC,
                            Wav2Vec2FeatureExtractor,
                            Wav2Vec2CTCTokenizer,
                            Wav2Vec2PreTrainedModel,
                            Wav2Vec2Processor,
                            EvalPrediction,
                            set_seed,)
#from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os
import shutil
import re
import json

from DeepDataMiningLearning.hfaudio.hfdata import create_vocabulary_from_data
from DeepDataMiningLearning.hfaudio.hfutil import valkey, TrustRemoteCode, logger

unk_token="[UNK]"
pad_token="[PAD]"
word_delimiter_token="|"

#https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/modeling_outputs.py
from transformers.modeling_outputs import TokenClassifierOutput
#Optin1: TokenClassifierOutput, Option2: MyClassifierOutput
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Dict, List, Optional, Union, Tuple



@dataclass
class MyModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AudioModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )
    adapter_attn_dim: int = field(
        default=16,
        metadata={
            "help": "The hidden dimension of the adapter layers that will be randomly initialized and trained. The higher the dimension, the more capacity is given to the adapter weights. Note that only the adapter weights are fine-tuned."
        },
    )

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
        self.task = task
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

        if self.task=="audio-classification":
            #self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
            #num_classes = len(id2label)
            self.classifier = MyClassificationHead(config=self.config)
            self.num_labels = self.config.num_labels
            self.init_weights()
        elif self.task=="audio-asr":
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
        if self.task == "audio-classification":
            if self.pooling_mode == 'mean':
                x = torch.mean(hidden_states, dim=1) #torch.Size([1, 768])
            elif self.pooling_mode == 'sum':
                x = torch.sum(hidden_states, dim=1)
            else: #'max'
                x = torch.max(hidden_states, dim=1)[0]
            logits = self.classifier(x)#(hidden_states) torch.Size([1, 45])
        elif self.task == "audio-asr":
            hidden_states = self.dropout(hidden_states)
            logits = self.lm_head_new(hidden_states)

        loss = None
        if labels is not None and self.task == "audio-classification":
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
        elif labels is not None and self.task=="audio-asr":
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

    #Create Processor
    processoroption1=False
    if processoroption1==True:
        processor = Wav2Vec2Processor.from_pretrained(model_checkpoint, cache_dir=mycache_dir,return_attention_mask=return_attention_mask)
        feature_extractor = processor.feature_extractor
    else:
        config = AutoConfig.from_pretrained(model_checkpoint, cache_dir=mycache_dir, trust_remote_code=TrustRemoteCode)#config.model_type:'wav2vec2' config.tokenizer_class=None
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
            #Wav2Vec2CTCTokenizer
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

def multilingual_tokenizer(model_name_or_path, tokenizer_name_or_path=None, mycache_dir=None, output_dir=None, datasets=None, target_column='text', target_language="en", overwrite_lang_vocab=True, overwrite_output_dir=False):
    
    if tokenizer_name_or_path=="":
        tokenizer_name_or_path = None
    # load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=mycache_dir,
        #token=data_args.token,
        trust_remote_code=TrustRemoteCode,
    )

    # 4. if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_kwargs = {}

    vocab_dict = {}
    if tokenizer_name_or_path: # and tokenizer_name_or_path is not None:
        # load vocabulary of other adapter languages so that new language can be appended
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            #token=data_args.token,
            trust_remote_code=TrustRemoteCode,
        )
        vocab_dict = tokenizer.vocab.copy()
        if tokenizer.target_lang is None:
            raise ValueError("Make sure to load a multi-lingual tokenizer with a set target language.")

        if target_language in tokenizer.vocab and not overwrite_lang_vocab:
            logger.info(
                "Adapter language already exists."
                " Skipping vocabulary creating. If you want to create a new vocabulary"
                f" for {target_language} make sure to add '--overwrite_lang_vocab'"
            )
        else:
            tokenizer_name_or_path = None

    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")
        if overwrite_output_dir and os.path.isfile(vocab_file):
            try:
                os.remove(vocab_file)
            except OSError:
                # in shared file-systems it might be the case that
                # two processes try to delete the vocab file at the some time
                pass
        
        if not os.path.isfile(vocab_file):
            os.makedirs(tokenizer_name_or_path, exist_ok=True)
            lang_dict = create_vocabulary_from_data(
                datasets,
                target_column=target_column,
                word_delimiter_token=word_delimiter_token,
                unk_token=unk_token,
                pad_token=pad_token,
            )

            # if we doing adapter language training, save
            # vocab with adpter language

            if target_language is not None:
                vocab_dict[target_language] = lang_dict

            # save vocab dict to be loaded into tokenizer
            with open(vocab_file, "w") as file:
                json.dump(vocab_dict, file)
        
        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        print(config.tokenizer_class) #None
        print(config.model_type) #wave2vec2
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
            "target_lang": target_language,
        }
    
    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        #token=data_args.token,
        trust_remote_code=TrustRemoteCode,
        **tokenizer_kwargs,
    )
    return tokenizer
    
def load_featureextractor_model(model_name, tokenizer, cache_dir, config, model_args=None, gradient_checkpointing=False):
    if model_args is None:
        model_args = AudioModelArguments(model_name_or_path=model_name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        #token=data_args.token,
        trust_remote_code=TrustRemoteCode,
    )
    # adapt config
    if not config:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            #token=data_args.token,
            trust_remote_code=TrustRemoteCode,
        )
    config.update(
        {
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "adapter_attn_dim": model_args.adapter_attn_dim,#add adapter for multi-language
        }
    )
    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        config=config,
        #token=data_args.token,
        trust_remote_code=TrustRemoteCode,
        ignore_mismatched_sizes=True,
    )

    # if attn adapter is defined, freeze all non-adapter weights
    if model.config.adapter_attn_dim is not None: #16
        model.init_adapter_layers()
        # first we freeze the whole base model
        model.freeze_base_model()

        # next we unfreeze all adapter layers
        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return model, processor, feature_extractor

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

#from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"
from safetensors.torch import save_file as safe_save_file
def save_adapterweights(model, target_language, output_dir):
    # make sure that adapter weights are saved seperately
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_language)
    adapter_file = os.path.join(output_dir, adapter_file)
    logger.info(f"Saving adapter weights under {adapter_file}...")
    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

def load_hfcheckpoint(checkpoint_dir, overwrite_output_dir=False):
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is None and len(os.listdir(checkpoint_dir)) > 0:
            raise ValueError(
                f"Output directory ({checkpoint_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

#jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn, facebook/mms-1b-all, jonatasgrosman/wav2vec2-large-xlsr-53-english, TencentGameMate/chinese-wav2vec2-base, facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-large-xlsr-53, anton-l/xtreme_s_xlsr_300m_minds14, facebook/wav2vec2-base-960h, "facebook/wav2vec2-base", ntu-spml/distilhubert


if __name__ == "__main__":
    from hfdata import gettestdata
    from datasets import DatasetDict
    model_name = "facebook/wav2vec2-xls-r-300m"
    model_args = AudioModelArguments(model_name_or_path=model_name)
    print(model_args.final_dropout)
    print(model_args)

    mycache_dir = "/DATA10T/Cache"
    output_dir = './output'

    dataset_name = "mozilla-foundation/common_voice_11_0"
    language = 'en'
    dataset_test = gettestdata(dataset_name, language=language, split="test", sampling_rate=16000, mycache_dir=mycache_dir, streaming=False, samples = 100, task_column="audio", target_column="sentence")
    raw_datasets = DatasetDict()
    raw_datasets["train"] = dataset_test

    tokenizer = multilingual_tokenizer(model_name, tokenizer_name_or_path="", mycache_dir=mycache_dir, output_dir=output_dir, datasets=raw_datasets, target_column="sentence", target_language="en", overwrite_lang_vocab=True, overwrite_output_dir=True)
    print(tokenizer)