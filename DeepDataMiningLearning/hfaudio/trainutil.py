import torch
from torch.optim import AdamW
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from transformers import default_data_collator

def get_datacollator(processor, task, padding=True):
    if processor:
        #processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        data_collator = MyDataCollatorWithPadding(processor=processor, padding=padding, task=task)
        #the padding tokens in the labels with -100 so that those tokens are not taken into account when computing the loss.
    else:
        data_collator = default_data_collator
    return data_collator

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
    #inputvaluekey: str = "input_values"
    task: Optional[str] = None
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        #input_features = [{"input_values": feature["input_values"]} for feature in features]
        #input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        onefeature=features[0]
        if "input_values" in onefeature.keys():
            inputvaluekey = "input_values"
        elif "input_features" in onefeature.keys():
            inputvaluekey = "input_features"
        input_features = [{inputvaluekey: feature[inputvaluekey]} for feature in features]

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

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
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
