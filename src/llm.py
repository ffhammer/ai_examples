import torch
from src.layers import SelfAttentionBlock, PositionalEncoding, Vocab
from torch import nn


class SmallLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_tokens,
        dim_size=256,
        n_blocks=4,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
