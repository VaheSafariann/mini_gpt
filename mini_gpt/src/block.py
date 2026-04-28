import torch.nn as nn
from mini_gpt.src.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x #Residual
        return x

