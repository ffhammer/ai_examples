import torch
import torch.nn as nn
import math


class Attention(nn.Module):

    def __init__(
        self,
        embed_size,
        num_heads,
        query_and_keys_head_size,
        values_head_size,
        dropout_prob,
    ) -> None:
        super().__init__()

        self.query_projection = nn.Linear(
            embed_size, num_heads * query_and_keys_head_size, bias=False
        )
        self.key_projection = nn.Linear(
            embed_size, num_heads * query_and_keys_head_size, bias=False
        )
        self.value_projection = nn.Linear(
            embed_size, num_heads * values_head_size, bias=False
        )
        self.output_projection = nn.Linear(num_heads * values_head_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.query_and_keys_head_size = query_and_keys_head_size
        self.values_head_size = values_head_size
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward(self, qs, ks, vs, mask=None):
        batch_size = qs.shape[0]

        assert ks.shape[:2] == vs.shape[:2]

        qs = self.query_projection(qs).view(
            batch_size, qs.shape[1], self.num_heads, self.query_and_keys_head_size
        )
        ks = self.key_projection(ks).view(
            batch_size, ks.shape[1], self.num_heads, self.query_and_keys_head_size
        )
        vs = self.value_projection(vs).view(
            batch_size, vs.shape[1], self.num_heads, self.values_head_size
        )

        attn_product = (
            qs.permute(0, 2, 1, 3)
            @ ks.permute(0, 2, 3, 1)
            / math.sqrt(self.query_and_keys_head_size)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, self.num_heads, qs.shape[1], ks.shape[1])

            attn_product.masked_fill_(mask == 0, -1e9)

        scores = nn.functional.softmax(attn_product, -1)

        scores = self.dropout(scores)

        res = scores @ vs.permute(0, 2, 1, 3)

        output = self.output_projection(
            res.transpose(1, 2).reshape(
                batch_size, -1, self.values_head_size * self.num_heads
            )
        )

        return output


class FasterSelfAttention(nn.Module):

    def __init__(self, embed_size, num_heads, attn_size, dropout_prob) -> None:
        super().__init__()

        self.qkv_projection = nn.Linear(
            embed_size, num_heads * attn_size * 3, bias=False
        )
        self.output_projection = nn.Linear(num_heads * attn_size, embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.attn_size = attn_size
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward(self, qs, mask=None):
        batch_size = qs.shape[0]

        qs, ks, vs = (
            self.qkv_projection(qs)
            .view(batch_size, qs.shape[1], self.num_heads * 3, self.attn_size)
            .split(self.num_heads, dim=2)
        )

        attn_product = (
            qs.permute(0, 2, 1, 3) @ ks.permute(0, 2, 3, 1) / math.sqrt(self.attn_size)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, self.num_heads, qs.shape[1], ks.shape[1])
            attn_product.masked_fill_(mask == 0, -1e9)

        scores = nn.functional.softmax(attn_product, -1)

        scores = self.dropout(scores)

        res = scores @ vs.permute(0, 2, 1, 3)

        output = self.output_projection(
            res.transpose(1, 2).reshape(batch_size, -1, self.attn_size * self.num_heads)
        )

        return output


class PointwiseFeedForward(nn.Module):

    def __init__(self, embed_size, dropout_prob, scaling_width=4) -> None:
        super().__init__()

        self.l1 = nn.Linear(embed_size, embed_size * scaling_width)
        self.l2 = nn.Linear(embed_size * scaling_width, embed_size)

        self.drop_out = nn.Dropout(dropout_prob)

    def forward(self, x):

        return self.l2(self.drop_out(nn.functional.relu(self.l1(x))))


class SubLayerLogic(nn.Module):

    def __init__(self, embed_size, dropout_prob) -> None:
        super().__init__()

        self.drop_out = nn.Dropout(dropout_prob)

        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, sublayer):

        return x + self.drop_out(sublayer(self.norm(x)))


class SelfAttentionBlock(nn.Module):

    def __init__(self, embed_size, num_heads, dropout_prob) -> None:
        super().__init__()

        self.attention = FasterSelfAttention(
            embed_size, num_heads, int(embed_size / num_heads), dropout_prob
        )
        self.att_sub = SubLayerLogic(embed_size, dropout_prob)

        self.feedforward = PointwiseFeedForward(embed_size, dropout_prob)
        self.feedforward_sub = SubLayerLogic(embed_size, dropout_prob)

    def forward(self, x, mask):

        self_attention = lambda x: self.attention(x, mask)

        x = self.att_sub(x, self_attention)
        x = self.feedforward_sub(x, self.feedforward)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Vocab(nn.Module):

    def __init__(self, embed_size, vocab_size) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)
