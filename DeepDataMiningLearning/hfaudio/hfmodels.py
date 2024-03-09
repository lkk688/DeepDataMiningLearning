from transformers import (AutoConfig, AutoTokenizer, pipeline, AutoProcessor,
                          AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForCTC, AutoModelForSpeechSeq2Seq,
                          SpeechEncoderDecoderModel,
                            Wav2Vec2Model,
                            Wav2Vec2Config,
                            Wav2Vec2ForCTC,
                            Wav2Vec2BertForCTC,
                            Wav2Vec2FeatureExtractor,
                            Wav2Vec2CTCTokenizer,
                            Wav2Vec2PreTrainedModel,
                            Wav2Vec2Processor,
                            Wav2Vec2BertProcessor,
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
from DeepDataMiningLearning.hfaudio.hfmodels_custom import MyWave2Vec2ClassificationCTC

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

def freezemodel(model, unfreezename="", freezename="", freeze_feature_encoder=True, freeze_base_model=True, use_adapter =False):
    model_num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")

    # if attn adapter is defined, freeze all non-adapter weights
    if use_adapter and model.config.adapter_attn_dim is not None: #16
        model.init_adapter_layers()
        #model.load_adapter("cmn-script_simplified")
        # first we freeze the whole base model
        model.freeze_base_model()
        # next we unfreeze all adapter layers
        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
    else:
        # freeze the convolutional waveform encoder
        # if hasattr(model, "freeze_feature_extractor"):
        #     model.freeze_feature_extractor()
        if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
            #model.freeze_feature_extractor()
            model.freeze_feature_encoder()
        if freeze_base_model and hasattr(model, "freeze_base_model"):
            model.freeze_base_model()
        modelparameters(model, unfreezename, freezename)
    return model

def modelparameters(model, unfreezename="", freezename="", unfreezehead=True):
    for name, param in model.named_parameters():
        need_unfreeze = False
        need_freeze = False
        if isinstance(unfreezename, list):
            for unfreezename_one in unfreezename:
                if unfreezename_one in name:
                    need_unfreeze = True
        elif unfreezename is not None:
            if unfreezename in name:
                need_unfreeze = True
        if isinstance(freezename, list):
            for freezename_one in freezename:
                if freezename_one in name:
                    need_freeze = True
        elif freezename is not None: 
            if freezename in name:
                need_freeze = True
        if 'head' in name and unfreezehead:
            need_unfreeze = True

        if need_unfreeze and not need_freeze:
            param.requires_grad = True
        elif need_freeze and not need_unfreeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
        print(name, param.requires_grad)
    # if unfreezename:
    #     for name, param in model.named_parameters():
    #         #if name.startswith(unfreezename): # choose whatever you like here
    #         if unfreezename in name:
    #             param.requires_grad = True
    #         elif unfreezehead and 'head' in name:
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    # if freezename:
    #     for name, param in model.named_parameters():
    #         if freezename in name:
    #             param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)


#https://github.com/huggingface/transformers/tree/main/src/transformers/models/wav2vec2
def loadmodel(model_checkpoint, custommodel=False, task="audio-classification", id2label=None, label2id=None, vocab_path=None, 
              pretrained="", cache_dir="", use_adapter=False, return_attention_mask=True):
    ignore_mismatched_sizes = custommodel #when loading model, ignore size 
    if vocab_path is not None:
        ignore_mismatched_sizes = True
    if cache_dir:
        mycache_dir = cache_dir
    elif os.environ.get('HF_HOME') is not None:
        mycache_dir = os.environ.get('HF_HOME')

    #option1
    #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #Option2
    #newtokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    #config.vocab_size:32

    #Create Processor with tokenizer
    processor = None
    if task =="audio-classification" or vocab_path is None: #processoroption1==True:
        #processor = Wav2Vec2Processor.from_pretrained(model_checkpoint, cache_dir=mycache_dir,return_attention_mask=return_attention_mask)
        processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir=mycache_dir, return_attention_mask=return_attention_mask)
        feature_extractor = processor.feature_extractor
    else:#task =="audio-asr" create own tokenizer
        config = AutoConfig.from_pretrained(model_checkpoint, cache_dir=mycache_dir, trust_remote_code=TrustRemoteCode)#config.model_type:'wav2vec2' config.tokenizer_class=None
        tokenizer_type = config.model_type if config.tokenizer_class is None else None #'wav2vec2' 'wav2vec2-bert'
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
        if 'bert' in tokenizer_type:
            processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        else:
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #get processor
    
    #Start to create the model
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
        elif "bert" in model_checkpoint:
            model = Wav2Vec2BertForCTC.from_pretrained(model_checkpoint, 
                                                       cache_dir = mycache_dir, 
                                                       vocab_size=len(processor.tokenizer), 
                                                       ignore_mismatched_sizes = ignore_mismatched_sizes,
                                                       add_adapter = use_adapter)
        else:
            #config = AutoConfig.from_pretrained(model_checkpoint)
            #print("Config vocab size", config.vocab_size)
            print(len(processor.tokenizer))
            #processor.tokenizer.add_tokens("_")
            #print(len(processor.tokenizer))
            model = AutoModelForCTC.from_pretrained(
                model_checkpoint, 
                cache_dir = mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=len(processor.tokenizer), #processor.tokenizer.vocab_size,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                feat_proj_dropout=0.0,
                mask_time_prob=0.05,
                layerdrop=0.1,
            )
            #The tokenizer's pad_token_id must be to define the model's pad_token_id or in the case of a CTC speech model also CTC's blank token
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            #token=data_args.token,
            cache_dir = mycache_dir,
            trust_remote_code=TrustRemoteCode,
            **tokenizer_kwargs,
        )
    except:
        tokenizer_kwargs = {
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
            "target_lang": target_language,
        }
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                tokenizer_name_or_path, #vocab_filepath,
                cache_dir = mycache_dir,
                trust_remote_code=TrustRemoteCode,
                **tokenizer_kwargs,
                )
    return tokenizer
    
def load_featureextractor_model(model_name, tokenizer, cache_dir, config, pretrained="", use_adapter=False):
    #if model_args is None:
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
            #"gradient_checkpointing": gradient_checkpointing,
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
    # model = Wav2Vec2ForCTC.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=cache_dir,
    #     config=config,
    #     #token=data_args.token,
    #     trust_remote_code=TrustRemoteCode,
    #     ignore_mismatched_sizes=True,
    #     target_lang="cmn-script_simplified",
    # )

    # if attn adapter is defined, freeze all non-adapter weights
    if use_adapter and model.config.adapter_attn_dim is not None: #16
        model.init_adapter_layers()
        #model.load_adapter("cmn-script_simplified")
        # first we freeze the whole base model
        model.freeze_base_model()
        # next we unfreeze all adapter layers
        adapter_weights = model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
    
    modelparameters(model, unfreezename="", freezename="wav2vec2_bert.encoder.layers")

    #pretrained
    starting_epoch = 0
    if pretrained:
        checkpoint = torch.load(pretrained, map_location='cpu')
        print("Pretrained epoch:", checkpoint['epoch'])
        starting_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['model_state_dict'])

    if "bert" in model_name:
        processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    else:
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return model, processor, feature_extractor, starting_epoch

#https://huggingface.co/docs/transformers/model_doc/speech-encoder-decoder
def load_SpeechEncoderDecoderModel(encoder_id = "facebook/wav2vec2-base-960h", decoder_id = "bert-base-uncased", cache_dir="", outputdir="", language=None, task="", freeze_feature_encoder=False, freeze_encoder=False):
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, cache_dir=cache_dir, trust_remote_code=TrustRemoteCode,)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id, cache_dir=cache_dir, trust_remote_code=TrustRemoteCode,)

    # Combine pre-trained encoder and pre-trained decoder to form a Seq2Seq model
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id, cache_dir=cache_dir)
    print(tokenizer.bos_token_id)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, feature_extractor, tokenizer

def load_featureextractor_seqmodel(model_name, tokenizer=None, cache_dir="", outputdir="", language=None, task="", freeze_feature_encoder=False, freeze_encoder=False):
    # if model_args is None:
    #     model_args = AudioModelArguments(model_name_or_path=model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        #revision=model_args.model_revision,
        #token=model_args.token,
        trust_remote_code=TrustRemoteCode,
    )#use_cache:True

    #config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": False}) #model_args.apply_spec_augment

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        #revision=model_args.model_revision,
        #token=data_args.token,
        trust_remote_code=TrustRemoteCode,
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True, #model_args.use_fast_tokenizer,
            #revision=model_args.model_revision,
            #token=model_args.token,
            trust_remote_code=TrustRemoteCode,
        )
    # create model
    #
    #AutoModelForSpeechSeq2Seq
    model = SpeechEncoderDecoderModel.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        #revision=model_args.model_revision,
        #token=model_args.token,
        trust_remote_code=TrustRemoteCode,
        ignore_mismatched_sizes=True,
    )
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    #if freeze_feature_encoder:
        #model.freeze_feature_encoder()

    # if freeze_encoder:
    #     model.freeze_encoder()
    #     model.model.encoder.gradient_checkpointing = False
    
    if language is not None and hasattr(tokenizer, 'set_prefix_tokens'):
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        #https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/tokenization_whisper.py
        tokenizer.set_prefix_tokens(language=language, task=None) #data_args.task
    
    feature_extractor.save_pretrained(outputdir)
    tokenizer.save_pretrained(outputdir)
    config.save_pretrained(outputdir)
    #AutoProcessor
    #processor = AutoProcessor.from_pretrained(outputdir)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #processor = Wav2Vec2Processor.from_pretrained(outputdir)

    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    for name, param in model.named_parameters():
        #if name.startswith("model.decoder.layers"): # choose whatever you like here
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    modelparameters(model, unfreezename="")
    #model.decoder.layers

    return model, processor, feature_extractor, forward_attention_mask

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
    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir) and not overwrite_output_dir:
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