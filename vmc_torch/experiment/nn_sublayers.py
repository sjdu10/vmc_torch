import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ShiftedSinh(nn.Module):
    """
    X-axis shifted sinh activation: f(x) = sinh(x + shift)
    Here shift is a learnable parameter.
    """

    def __init__(self, init_shift: float = 0.0):
        super().__init__()
        # shift 作为可学习参数
        self.shift = nn.Parameter(torch.tensor(init_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x + self.shift)


class ShiftedSinhY(nn.Module):
    """
    Y-axis shifted sinh activation: f(x) = sinh(x) + shift
    Here shift is a learnable parameter.
    """

    def __init__(self, init_shift: float = 0.0):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(init_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x) + self.shift


class ShiftedSinhYFixed(nn.Module):
    """
    Y-axis shifted sinh activation: f(x) = sinh(x) + shift
    Here shift is a fixed parameter.
    """

    def __init__(self, shift: float = 1.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinh(x) + self.shift


class PositionwiseFeedForward(nn.Module):
    """Position-wise 2-layer feed-forward MLP layer."""

    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()
        # assuming d_in == d_out
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, **mha_kwargs):
        super().__init__()
        # instantiate the real MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **mha_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ):
        # internally use x for (query, key, value)
        return self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )


class SelfAttn_block_pos(nn.Module):
    """ Self-attention block with positional encoding"""
    def __init__(
        self, n_site, num_classes, embed_dim, attention_heads, dtype=torch.float32
    ):
        super(SelfAttn_block_pos, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embed_dim)
        # Learnable positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(n_site, embed_dim) / embed_dim**0.5
        )
        # Self-attention block
        self.self_attention = SelfAttention(
            embed_dim=embed_dim, num_heads=attention_heads
        )

        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.positional_embedding.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)
        embedded = embedded + self.positional_embedding

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(embedded)

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        return attn_output


class SelfAttn_block(nn.Module):
    """ Plain self-attention block with one-hot embedding and layer norm"""
    def __init__(
        self, n_site, num_classes, embedding_dim, attention_heads, dtype=torch.float32
    ):
        super(SelfAttn_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True
        )

        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        return attn_output


class SelfAttn_MLP(nn.Module):
    """ Self-attention block followed by position-wise feed-forward network (MLP)"""
    def __init__(
        self,
        n_site,
        num_classes,
        embed_dim,
        attention_heads,
        dtype=torch.float32,
        layer_norm=False,
        position_wise_mlp=True,
        positional_encoding=True,
    ):
        super(SelfAttn_MLP, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embed_dim)
        # Learnable positional embedding
        self.positional_embedding = (
            nn.Parameter(torch.randn(n_site, embed_dim) / embed_dim**0.5)
            if positional_encoding
            else None
        )
        # Self-attention block
        self.self_attention = SelfAttention(
            embed_dim=embed_dim, num_heads=attention_heads
        )
        # Position-wise feed-forward network
        self.mlp = (
            PositionwiseFeedForward(d_in=embed_dim, d_hid=embed_dim * 2)
            if position_wise_mlp
            else lambda x: x
        )

        self.dtype = dtype
        self.layer_norm = layer_norm
        self.position_wise_mlp = position_wise_mlp
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.positional_embedding.to(dtype=dtype) if positional_encoding else None
        self.mlp.to(dtype=dtype) if position_wise_mlp else None

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)
        embedded = (
            embedded + self.positional_embedding
            if self.positional_embedding is not None
            else embedded
        )

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(embedded)

        # Step 4: Residual connection and layer normalization
        # If layer_norm is True, apply layer normalization
        if self.layer_norm:
            attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])
        else:
            attn_output = attn_output + embedded

        # Step 5: Pass through the position-wise feed-forward network with residual connection
        if self.position_wise_mlp:
            if self.layer_norm:
                attn_output = F.layer_norm(
                    self.mlp(attn_output) + attn_output, attn_output.size()[1:]
                )
            else:
                attn_output = self.mlp(attn_output) + attn_output
        else:
            pass

        return attn_output


class SelfAttn_FFNN_block(nn.Module):
    """ Self-attention block followed by a final feed-forward network to get fixed-length output"""
    def __init__(
        self,
        n_site,
        num_classes,
        embedding_dim,
        attention_heads,
        nn_hidden_dim,
        output_dim,
        dtype=torch.float32,
    ):
        super(SelfAttn_FFNN_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True
        )

        # Final feed-forward network to project the flattened attention output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(embedding_dim * n_site, nn_hidden_dim),
            nn.LeakyReLU(), #XXX: perhaps use GELU?
            nn.Linear(nn_hidden_dim, output_dim),
        )
        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        # Step 5: Reshape the output and flatten it
        flattened_output = attn_output.view(-1)

        # Step 6: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output

class SelfAttn_FFNN_pos_block(nn.Module):
    """ Self-attention block followed by a final feed-forward network to get fixed-length output"""
    def __init__(
        self,
        n_site,
        num_classes,
        embedding_dim,
        attention_heads,
        nn_hidden_dim,
        output_dim,
        dtype=torch.float32,
    ):
        super(SelfAttn_FFNN_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Learnable positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(n_site, embedding_dim) / embedding_dim**0.5
        )

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True
        )

        # Final feed-forward network to project the flattened attention output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(embedding_dim * n_site, nn_hidden_dim),
            nn.LeakyReLU(), #XXX: perhaps use GELU?
            nn.Linear(nn_hidden_dim, output_dim),
        )
        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.positional_embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)
        embedded = embedded + self.positional_embedding

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        # Step 5: Reshape the output and flatten it
        flattened_output = attn_output.view(-1)

        # Step 6: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output


class StackedSelfAttn_FFNN(nn.Module):
    """ Stacked self-attention blocks followed by a final feed-forward network to get fixed-length output"""
    def __init__(
        self,
        n_site,
        num_classes,
        output_dim,
        nn_hidden_dim=128,
        num_layers=1,
        embedding_dim=8,
        d_inner=16,
        attention_heads=2,
        use_positional_encoding=True,
        dtype=torch.float32,
    ):
        super(StackedSelfAttn_FFNN, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, self.embedding_dim)

        # Stacking transformer blocks (Self-attention + FFNN) (Nonlinearities in softmax and 2-layer MLP)
        self.transformer_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": nn.MultiheadAttention(
                            embed_dim=self.embedding_dim,
                            num_heads=attention_heads,
                            batch_first=True,
                        ),
                        "layer_norm1": nn.LayerNorm(self.embedding_dim),
                        "ffn": PositionwiseFeedForward(
                            d_in=self.embedding_dim, d_hid=d_inner
                        ),
                        "layer_norm2": nn.LayerNorm(self.embedding_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Learnable positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(n_site, self.embedding_dim) / self.embedding_dim**0.5
        )
        self.use_positional_encoding = use_positional_encoding

        # Final feed-forward network to project the flattened output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim * n_site, nn_hidden_dim),
            nn.GELU(),
            nn.Linear(nn_hidden_dim, output_dim),
        )
        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.transformer_blocks.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(
            self.dtype
        )

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        if self.use_positional_encoding:
            embedded = embedded + self.positional_embedding

        # Step 3: Pass through the stacked transformer blocks
        x = embedded
        for block in self.transformer_blocks:
            # Self-attention block
            attn_output, _ = block["attention"](x, x, x)
            x = block["layer_norm1"](
                x + attn_output
            )  # Residual connection and normalization

            # Position-wise feed-forward block and residual connection and normalization
            x = block["layer_norm2"](x + block["ffn"](x))  

        # Step 4: Reshape the output and flatten it
        flattened_output = x.view(-1)  # Flatten along the last dimension

        # Step 5: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output
