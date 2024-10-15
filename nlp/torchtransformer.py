import torch
from torch import nn

def test_torchtransformer():
    #https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    #https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """
    src = torch.rand((10, 32, 512)) #(seq, batch, feature)
    tgt = torch.rand((20, 32, 512)) #(seq, batch, feature)
    
    #The tgt tensor is necessary during training to help the model learn the correct output sequences. 
    # This process is known as teacher forcing, where the actual target sequence is fed into the model to guide its learning.
    out = transformer_model(src, tgt)
    print(out.shape) #[20, 32, 512]
    
    #During inference, you don’t provide the entire tgt sequence upfront 
    # Autoregressive decoding.
    src = torch.rand((10, 32, 512))  # (sequence length, batch size, feature size)

    # Initialize the target sequence with the start-of-sequence token
    tgt = torch.zeros((1, 32, 512))  # (sequence length, batch size, feature size)
    #Start with an initial token: Begin with a special start-of-sequence token (<sos>).
    
    max_length = 20
    
    # Autoregressive decoding loop
    for i in range(1, max_length):
        #Generate one token at a time: Feed the src and the tokens generated so far into the model to predict the next token.
        output = transformer_model(src, tgt)
        next_token = output[-1, :, :]  # Get the last token's prediction
        #Append the predicted token: Add the predicted token to the sequence and repeat the process until you reach an end-of-sequence token (<eos>) or the desired sequence length.
        tgt = torch.cat((tgt, next_token.unsqueeze(0)), dim=0)  # Append the predicted token

    # The final output sequence
    predicted_sequence = tgt
    print(predicted_sequence.shape)

if __name__ == "__main__":
    test_torchtransformer()