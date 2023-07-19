#ref: https://github.com/shahrukhx01/multitask-learning-transformers

import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO)

from datasets import load_dataset
import transformers
#STS-B: A two-sentece textual similarity scoring task. (Prediction is a real number between 1 and 5)
#RTE: A two-sentence natural language entailment task. (Prediction is one of two classes)
#Commonsense QA: A multiple-choice question-answering task. (Each example consists of 5 seperate text inputs, prediction is which one of the 5 choices is correct)

#https://huggingface.co/datasets/glue
dataset_dict = {
    "stsb": load_dataset('glue', name="stsb"),
    "rte": load_dataset('glue', name="rte"),
    "commonsense_qa": load_dataset('commonsense_qa'),
}

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

max_length = 128

def convert_to_stsb_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rte_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_commonsense_qa_features(example_batch):
    num_examples = len(example_batch["question"])
    num_choices = len(example_batch["choices"][0]["text"])
    features = {}
    for example_i in range(num_examples):
        choices_inputs = tokenizer.batch_encode_plus(
            list(zip(
                [example_batch["question"][example_i]] * num_choices,
                example_batch["choices"][example_i]["text"],
            )),
            max_length=max_length, pad_to_max_length=True,
        )
        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)
    labels2id = {char: i for i, char in enumerate("ABCDE")}
    # Dummy answers for test
    if example_batch["answerKey"][0]:
        features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
    else:
        features["labels"] = [0] * num_examples    
    return features

if __name__ == "__main__":
    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()
    
    model_name = "roberta-base"
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "stsb": transformers.AutoModelForSequenceClassification,
            "rte": transformers.AutoModelForSequenceClassification,
            "commonsense_qa": transformers.AutoModelForMultipleChoice,
        },
        model_config_dict={
            "stsb": transformers.AutoConfig.from_pretrained(model_name, num_labels=1),
            "rte": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
            "commonsense_qa": transformers.AutoConfig.from_pretrained(model_name),
        },
    )
    if model_name.startswith("roberta-"):
        print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["stsb"].roberta.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["rte"].roberta.embeddings.word_embeddings.weight.data_ptr())
        print(multitask_model.taskmodels_dict["commonsense_qa"].roberta.embeddings.word_embeddings.weight.data_ptr())
    else:
        print("Exercise for the reader: add a check for other model architectures =)")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    convert_func_dict = {
        "stsb": convert_to_stsb_features,
        "rte": convert_to_rte_features,
        "commonsense_qa": convert_to_commonsense_qa_features,
    }

    columns_dict = {
        "stsb": ['input_ids', 'attention_mask', 'labels'],
        "rte": ['input_ids', 'attention_mask', 'labels'],
        "commonsense_qa": ['input_ids', 'attention_mask', 'labels'],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))