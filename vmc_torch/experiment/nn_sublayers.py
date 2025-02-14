import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    

class SelfAttn_FFNN_block(nn.Module):
    def __init__(self, n_site, num_classes, embedding_dim, attention_heads, nn_hidden_dim, output_dim, dtype=torch.float32):
        super(SelfAttn_FFNN_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)

        # Final feed-forward network to project the flattened attention output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(embedding_dim * n_site, nn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden_dim, output_dim)
        )
        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(self.dtype)

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(embedded, embedded, embedded, need_weights=False)

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        # Step 5: Reshape the output and flatten it
        flattened_output = attn_output.view(-1)

        # Step 6: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output

class SelfAttn_FFNN_proj_block(nn.Module):
    def __init__(self, n_site, num_classes, embedding_dim, attention_heads, nn_hidden_dim, proj_dim, output_dim, dtype=torch.float32):
        super(SelfAttn_FFNN_proj_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True)

        # Final feed-forward network to project the flattened attention output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(embedding_dim * n_site, nn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden_dim, proj_dim),
            nn.LeakyReLU(),
            nn.Linear(proj_dim, output_dim)
        )

        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(self.dtype)

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        # Step 3: Pass through the self-attention block
        attn_output, _ = self.self_attention(embedded, embedded, embedded, need_weights=False)

        # Step 4: Residual connection and layer normalization
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        # Step 5: Reshape the output and flatten it
        flattened_output = attn_output.view(-1)

        # Step 6: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output




class StackedSelfAttn_FFNN(nn.Module):
    def __init__(self, n_site, num_classes, output_dim, num_attention_blocks=1, embedding_dim=8, d_inner=16, nn_hidden_dim=128, attention_heads=2, dropout=0.0, dtype=torch.float32):
        super(StackedSelfAttn_FFNN, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, self.embedding_dim)

        # Stacking transformer blocks (Self-attention + FFNN)
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=attention_heads, batch_first=True),
                "layer_norm1": nn.LayerNorm(self.embedding_dim),
                "ffn": PositionwiseFeedForward(d_in=self.embedding_dim, d_hid=d_inner, dropout=dropout),
                "layer_norm2": nn.LayerNorm(self.embedding_dim)
            }) for _ in range(num_attention_blocks)
        ])

        # Final feed-forward network to project the flattened output to `output_dim`
        self.final_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim * n_site, nn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden_dim, output_dim)
        )
        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.transformer_blocks.to(dtype=dtype)
        self.final_ffn.to(dtype=dtype)

    def forward(self, input_seq):
        # Step 1: One-hot encode the input sequence
        one_hot_encoded = F.one_hot(input_seq.long(), num_classes=self.num_classes).to(self.dtype)

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded)

        # Step 3: Pass through the stacked transformer blocks
        x = embedded
        for block in self.transformer_blocks:
            # Self-attention block
            attn_output, _ = block["attention"](x, x, x)
            x = block["layer_norm1"](x + attn_output)  # Residual connection and normalization

            # Feed-forward block
            x = block["ffn"](x)  # Includes residual connection and normalization

        # Step 4: Reshape the output and flatten it
        flattened_output = x.view(-1)  # Flatten along the last dimension

        # Step 5: Pass through the final feed-forward network to get fixed-length output
        final_output = self.final_ffn(flattened_output)

        return final_output