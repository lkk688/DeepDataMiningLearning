from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2BertPreTrainedModel, Wav2Vec2BertModel
from DeepDataMiningLearning.hfaudio.hfutil import valkey, TrustRemoteCode

@dataclass
class MyModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

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
            self.wav2vec2 = Wav2Vec2Model(config, cache_dir = mycache_dir)
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
    ) -> Union[Tuple, MyModelOutput]:
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

class MyWave2Vec2BertClassificationCTC(Wav2Vec2BertPreTrainedModel):
    def __init__(self, config, basemodel_name=None, task="audio-classification", mycache_dir=None, pooling_mode='mean', target_lang: Optional[str] = None):
        super().__init__(config)

        self.pooling_mode = pooling_mode#['mean', 'sum', 'max']
        self.task = task
        self.has_weights = False
        if basemodel_name:
            self.wav2vec2_bert = Wav2Vec2BertModel.from_pretrained(basemodel_name,cache_dir = mycache_dir, trust_remote_code=TrustRemoteCode) 
            self.has_weights = True
            self.config = config
        elif config:
            self.wav2vec2_bert = Wav2Vec2BertModel(config, cache_dir = mycache_dir)
            self.config = config
        else:
            print("Error in MyWave2Vec2Classification init!")
        
        self.target_lang = target_lang
        print(self.wav2vec2_bert.feature_extractor) #Wav2Vec2FeatureEncoder

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
            self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
            if not self.has_weights:
                # Initialize weights and apply final processing
                self.post_init()
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MyModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
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
