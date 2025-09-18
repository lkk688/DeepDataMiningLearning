#modified from mybertmodel.py under nlp folder
import os
import math
from transformers import BertConfig #BertModel

from transformers.utils.generic import ModelOutput
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from transformers import AutoTokenizer, BertForQuestionAnswering
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
#ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
#from transformers.models.bert.modeling_bert import BertPreTrainedModel
class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    #load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class myBertModel(BertPreTrainedModel):
    """
    BERT model implementation based on the original paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    
    This model consists of:
    1. Embedding layer (token, position, token type embeddings)
    2. Encoder with multiple transformer layers
    3. Optional pooler layer for sentence-level representations
    
    Architecture:
    Input -> Embeddings -> Transformer Layers -> [Pooler] -> Output
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Embedding layer combines token, position and token type embeddings
        self.embeddings = BertEmbeddings(config)
        
        # Encoder contains multiple transformer layers (self-attention + feed-forward)
        self.encoder = BertEncoder(config)

        # Pooler applies a linear layer + tanh to the first token ([CLS]) for sentence-level tasks
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Returns the model's token embedding layer for weight tying"""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Sets the model's token embedding layer for weight tying"""
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes attention heads in the model
        Args:
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,              # Shape: [batch_size, seq_length]
        attention_mask: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        token_type_ids: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        position_ids: Optional[torch.Tensor] = None,           # Shape: [batch_size, seq_length]
        head_mask: Optional[torch.Tensor] = None,              # Shape: [num_hidden_layers, batch_size, num_heads, seq_length, seq_length]
        inputs_embeds: Optional[torch.Tensor] = None,          # Shape: [batch_size, seq_length, hidden_size]
        encoder_hidden_states: Optional[torch.Tensor] = None,  # Shape: [batch_size, encoder_seq_length, hidden_size]
        encoder_attention_mask: Optional[torch.Tensor] = None, # Shape: [batch_size, encoder_seq_length]
        past_key_values: Optional[List[torch.FloatTensor]] = None, # For efficient decoding
        use_cache: Optional[bool] = None,                      # Whether to use cached key/values
        output_attentions: Optional[bool] = None,              # Whether to return attention weights
        output_hidden_states: Optional[bool] = None,           # Whether to return all hidden states
        return_dict: Optional[bool] = None,                    # Whether to return a ModelOutput object
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Forward pass of the BERT model
        
        Transformer flow:
        1. Convert input tokens to embeddings
        2. Process through multiple transformer layers
        3. Optionally pool the output for sentence-level tasks
        
        Returns:
            - sequence_output: Token-level output for all tokens (last hidden state)
            - pooled_output: Sentence-level output (first token transformed)
            - Optional: hidden states, attentions, etc. based on parameters
        """
        # Set default values for configuration options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Cache is only used in decoder models, BERT is an encoder-only model
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # Validate inputs - either input_ids or inputs_embeds must be provided
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # Get input shape from input_ids
            input_shape = input_ids.size()  # [batch_size, seq_length]
        elif inputs_embeds is not None:
            # Get input shape from inputs_embeds
            input_shape = inputs_embeds.size()[:-1]  # [batch_size, seq_length]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Extract batch size and sequence length
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Calculate past key values length for position embeddings
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Create attention mask if not provided
        if attention_mask is None:
            # Default: attend to all tokens
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # Create token type ids if not provided
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                # Use the model's buffered token type ids
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # Default: all tokens belong to the same segment (0)
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Prepare attention mask for multi-head attention
        # Convert from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
        # with 1.0 for tokens to attend to and 0.0 for tokens to ignore
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare encoder attention mask for cross-attention in decoder models
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                # Default: attend to all encoder tokens
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            # Convert encoder attention mask to format needed for cross-attention
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if provided
        # 1.0 in head_mask indicates we keep the head, 0.0 means we prune it
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Step 1: Get embeddings from token IDs or direct embeddings
        # Output shape: [batch_size, seq_length, hidden_size]
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # Step 2: Process through transformer encoder layers
        # Each layer applies:
        # - Multi-head self-attention
        # - Add & Norm
        # - Feed-forward network
        # - Add & Norm
        encoder_outputs = self.encoder(
            embedding_output,                                # [batch_size, seq_length, hidden_size]
            attention_mask=extended_attention_mask,          # [batch_size, 1, 1, seq_length]
            head_mask=head_mask,                             # [num_layers, batch_size, num_heads, seq_length, seq_length]
            encoder_hidden_states=encoder_hidden_states,     # For cross-attention in decoder models
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the final hidden states from encoder
        # Shape: [batch_size, seq_length, hidden_size]
        sequence_output = encoder_outputs[0]
        
        # Step 3: Apply pooling to get sentence representation if needed
        # Takes the [CLS] token (first token) and applies a linear layer + tanh
        # Shape: [batch_size, hidden_size]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # Return outputs based on return_dict flag
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # Return a structured output object with all requested outputs
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,       # [batch_size, seq_length, hidden_size]
            pooler_output=pooled_output,             # [batch_size, hidden_size]
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Output type for token classification models.
    
    Args:
        loss: Optional classification loss
        logits: Classification logits for each token
        hidden_states: Optional hidden states from all layers
        attentions: Optional attention weights from all layers
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class myBertForTokenClassification(BertPreTrainedModel):
    """
    BERT model for token-level classification tasks like Named Entity Recognition (NER)
    
    Architecture:
    Input -> BERT -> Dropout -> Classification Head -> Output
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # BERT model without pooling (we need token-level outputs)
        self.bert = myBertModel(config, add_pooling_layer=False)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,              # Shape: [batch_size, seq_length]
        attention_mask: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        token_type_ids: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        position_ids: Optional[torch.Tensor] = None,           # Shape: [batch_size, seq_length]
        head_mask: Optional[torch.Tensor] = None,              # Shape: [num_hidden_layers, batch_size, num_heads, seq_length, seq_length]
        inputs_embeds: Optional[torch.Tensor] = None,          # Shape: [batch_size, seq_length, hidden_size]
        labels: Optional[torch.Tensor] = None,                 # Shape: [batch_size, seq_length]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Forward pass for token classification
        
        Process:
        1. Get token-level representations from BERT
        2. Apply dropout for regularization
        3. Apply classification head to each token
        4. Calculate loss if labels are provided
        
        Returns:
            - loss (optional): Classification loss
            - logits: Classification scores for each token
            - hidden_states (optional): All hidden states
            - attentions (optional): All attention weights
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get token-level outputs from BERT
        # sequence_output shape: [batch_size, seq_length, hidden_size]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the sequence output (token representations)
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        
        # Apply dropout for regularization
        sequence_output = self.dropout(sequence_output)  # [batch_size, seq_length, hidden_size]
        
        # Apply classification head to each token
        logits = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss (ignore padding tokens)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Return outputs based on return_dict flag
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # Return a structured output object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,                      # [batch_size, seq_length, num_labels]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Output type for sequence classification models.
    
    Args:
        loss: Optional classification loss
        logits: Classification logits for the sequence
        hidden_states: Optional hidden states from all layers
        attentions: Optional attention weights from all layers
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class myBertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model for sequence-level classification tasks like sentiment analysis
    
    Architecture:
    Input -> BERT -> Pooler -> Dropout -> Classification Head -> Output
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # BERT model with pooling (we need sentence-level representation)
        self.bert = myBertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,              # Shape: [batch_size, seq_length]
        attention_mask: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        token_type_ids: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        position_ids: Optional[torch.Tensor] = None,           # Shape: [batch_size, seq_length]
        head_mask: Optional[torch.Tensor] = None,              # Shape: [num_hidden_layers, batch_size, num_heads, seq_length, seq_length]
        inputs_embeds: Optional[torch.Tensor] = None,          # Shape: [batch_size, seq_length, hidden_size]
        labels: Optional[torch.Tensor] = None,                 # Shape: [batch_size]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification
        
        Process:
        1. Get sentence-level representation from BERT (pooled output)
        2. Apply dropout for regularization
        3. Apply classification head to get logits
        4. Calculate loss if labels are provided
        
        Returns:
            - loss (optional): Classification loss
            - logits: Classification scores for the sequence
            - hidden_states (optional): All hidden states
            - attentions (optional): All attention weights
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get outputs from BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the pooled output (sentence representation)
        pooled_output = outputs[1]  # [batch_size, hidden_size]
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)  # [batch_size, hidden_size]
        
        # Apply classification head
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # Regression task
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # Classification task
                    self.config.problem_type = "single_label_classification"
                else:
                    # Multi-label classification
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # Return outputs based on return_dict flag
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # Return a structured output object
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,                      # [batch_size, num_labels]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class myBertForQuestionAnswering(BertPreTrainedModel):
    """
    BERT model for question answering tasks like SQuAD
    
    Architecture:
    Input -> BERT -> QA Head (Linear) -> Start/End Logits
    
    The model predicts the start and end positions of the answer span within the context.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # BERT model without pooling (we need token-level outputs)
        self.bert = myBertModel(config, add_pooling_layer=False)
        
        # QA head: a linear layer that predicts start and end positions
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # 2 outputs: start and end positions
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,              # Shape: [batch_size, seq_length]
        attention_mask: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        token_type_ids: Optional[torch.Tensor] = None,         # Shape: [batch_size, seq_length]
        position_ids: Optional[torch.Tensor] = None,           # Shape: [batch_size, seq_length]
        head_mask: Optional[torch.Tensor] = None,              # Shape: [num_hidden_layers, batch_size, num_heads, seq_length, seq_length]
        inputs_embeds: Optional[torch.Tensor] = None,          # Shape: [batch_size, seq_length, hidden_size]
        start_positions: Optional[torch.Tensor] = None,        # Shape: [batch_size]
        end_positions: Optional[torch.Tensor] = None,          # Shape: [batch_size]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        """
        Forward pass for question answering
        
        Process:
        1. Get token-level representations from BERT
        2. Apply QA head to predict start and end positions
        3. Calculate loss if start_positions and end_positions are provided
        
        Returns:
            - loss (optional): Combined loss for start and end positions
            - start_logits: Scores for each token being the start of the answer
            - end_logits: Scores for each token being the end of the answer
            - hidden_states (optional): All hidden states
            - attentions (optional): All attention weights
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get token-level outputs from BERT
        outputs = self.bert(
            input_ids,                          # [batch_size, seq_length]
            attention_mask=attention_mask,      # [batch_size, seq_length]
            token_type_ids=token_type_ids,      # [batch_size, seq_length]
            position_ids=position_ids,          # None or [batch_size, seq_length]
            head_mask=head_mask,                # None or [num_layers, batch_size, num_heads, seq_length, seq_length]
            inputs_embeds=inputs_embeds,        # None or [batch_size, seq_length, hidden_size]
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the sequence output (token representations)
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        
        # Apply QA head to predict start and end positions
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_length, 2]
        
        # Split logits into start and end predictions
        start_logits, end_logits = logits.split(1, dim=-1)  # Each: [batch_size, seq_length, 1]
        start_logits = start_logits.squeeze(-1).contiguous()  # [batch_size, seq_length]
        end_logits = end_logits.squeeze(-1).contiguous()  # [batch_size, seq_length]
        
        # Calculate loss if start_positions and end_positions are provided
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Handle multi-GPU case where positions have an extra dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
                
            # Sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)  # seq_length
            start_positions = start_positions.clamp(0, ignored_index)  # Ensure valid indices
            end_positions = end_positions.clamp(0, ignored_index)  # Ensure valid indices
            
            # Calculate loss using cross entropy
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            # Total loss is the average of start and end losses
            total_loss = (start_loss + end_loss) / 2
        
        # Return outputs based on return_dict flag
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # Return a structured output object
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,          # [batch_size, seq_length]
            end_logits=end_logits,              # [batch_size, seq_length]
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = myBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class myBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = myBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels) #768,2

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids, #[1, 16]
            attention_mask=attention_mask, #[1, 16]
            token_type_ids=token_type_ids, #[1, 16] [0,0, ..1,1,]
            position_ids=position_ids, #None
            head_mask=head_mask, #None
            inputs_embeds=inputs_embeds,#None
            output_attentions=output_attentions,#None
            output_hidden_states=output_hidden_states,#None
            return_dict=return_dict, #True
        )

        sequence_output = outputs[0] #last hidden output[1, 16, 768]

        logits = self.qa_outputs(sequence_output) #[1, 16, 2]
        start_logits, end_logits = logits.split(1, dim=-1) #[1, 16, 1] [1, 16, 1]
        start_logits = start_logits.squeeze(-1).contiguous() #[1, 16]
        end_logits = end_logits.squeeze(-1).contiguous() #[1, 16]

        total_loss = None
        if start_positions is not None and end_positions is not None: #training mode
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict: #not run
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def loadsave_model(modelname="deepset/bert-base-cased-squad2"):
    modeloutputpath=os.path.join('./output', modelname)
    os.makedirs(modeloutputpath, exist_ok=True)
    model = BertForQuestionAnswering.from_pretrained(modelname)
    model.save_pretrained(modeloutputpath)

    modelfilepath=os.path.join(modeloutputpath, 'savedmodel.pth')
    torch.save({
            'model_state_dict': model.state_dict()
        }, modelfilepath)
    
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer.save_pretrained(modeloutputpath)

    
def load_QAbertmodel(rootpath='./output', modelname="deepset/bert-base-cased-squad2"):
    #loadsave_model()
    
    modelpath=os.path.join(rootpath, modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)

    configuration = BertConfig()
    configuration.vocab_size = 28996
    mybertqa = myBertForQuestionAnswering(config=configuration)
    modelfilepath=os.path.join(modelpath, 'savedmodel.pth')
    checkpoint = torch.load(modelfilepath, map_location='cpu')
    mybertqa.load_state_dict(checkpoint['model_state_dict'], strict=False)
    embedding_size = mybertqa.get_input_embeddings().weight.shape[0]
    print("Embeeding size:", embedding_size) #28996
    return mybertqa, tokenizer

def testBertQuestionAnswering():
    mybertqa, tokenizer = load_QAbertmodel()
    #tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = mybertqa(**inputs)

    answer_start_index = outputs.start_logits.argmax() #[1, 16] argmax ->12
    answer_end_index = outputs.end_logits.argmax() #->14

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    result=tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    print(result)


def demonstrate_bert_models():
    """
    Demonstrates how to use the BERT models for different tasks
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Example 1: Base BERT model
    print("Example 1: Base BERT model")
    config = BertConfig()
    model = myBertModel(config)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")  # [1, seq_len, hidden_size]
    print(f"Pooler output shape: {outputs.pooler_output.shape}")  # [1, hidden_size]
    print()
    
    # Example 2: Token Classification (e.g., NER)
    print("Example 2: Token Classification")
    config = BertConfig()
    config.num_labels = 9  # For NER: O, B-PER, I-PER, B-ORG, I-ORG, etc.
    token_classifier = myBertForTokenClassification(config)
    inputs = tokenizer("John Smith works at Microsoft", return_tensors="pt")
    outputs = token_classifier(**inputs)
    print(f"Token classification logits shape: {outputs.logits.shape}")  # [1, seq_len, num_labels]
    print()
    
    # Example 3: Sequence Classification (e.g., Sentiment Analysis)
    print("Example 3: Sequence Classification")
    config = BertConfig()
    config.num_labels = 2  # Binary classification: positive/negative
    seq_classifier = myBertForSequenceClassification(config)
    inputs = tokenizer("I love this movie!", return_tensors="pt")
    outputs = seq_classifier(**inputs)
    print(f"Sequence classification logits shape: {outputs.logits.shape}")  # [1, num_labels]
    print()
    
    # Example 4: Question Answering
    print("Example 4: Question Answering")
    config = BertConfig()
    qa_model = myBertForQuestionAnswering(config)
    question, text = "Who was Jim Henson?", "Jim Henson was a puppeteer"
    inputs = tokenizer(question, text, return_tensors="pt")
    outputs = qa_model(**inputs)
    print(f"Start logits shape: {outputs.start_logits.shape}")  # [1, seq_len]
    print(f"End logits shape: {outputs.end_logits.shape}")  # [1, seq_len]
    
    # Get answer span
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()

    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    bertmodel = myBertModel(configuration)
    #model = BertModel.from_pretrained("bert-base-uncased")
    bertmodel=bertmodel.from_pretrained("bert-base-uncased")

    # Accessing the model configuration
    configuration = bertmodel.config
    print(configuration.hidden_size)
    print(configuration.num_labels)

    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = bertmodel(**inputs)
    last_hidden_states = outputs.last_hidden_state #torch.Size([1, 8, 768])

    loadsave_model()
    testBertQuestionAnswering()

    demonstrate_bert_models()


    