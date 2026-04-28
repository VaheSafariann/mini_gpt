import torch
import torch.nn as nn
from mini_gpt.src.block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim)
        self.position_embedding = nn.Embedding(
            cfg.context_len, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(
                cfg.embed_dim,
                cfg.n_heads,
                cfg.dropout)
                for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(
            cfg.embed_dim, cfg.vocab_size,bias=False
            )

    def forward(self, x): #x = [B,T]
        B,T = x.shape #Batch,con_len(T),embed_dim(C)
        positions = torch.arange(
            T, device = x.device)#[0,1,,255] = T
        x = (self.token_embedding(x)#[B,T]->[B,T,embed_dim]
        + self.position_embedding(positions))#[T]->[T,embed_dim]
        #.+.=[B,T,embed_dim]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)#nuyny
        x = self.lm_head(x)
        return x #[B,T,vocab_size]