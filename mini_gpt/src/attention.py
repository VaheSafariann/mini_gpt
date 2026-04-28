import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        '''
        B = BatchSize = 16
        T = ContextLen = 128
        C = EmbedDim = 192
        '''
        Q  = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        B, T, C = Q.shape

        Q = Q.reshape(B, T, self.n_heads, self.d_head)
        K = K.reshape(B, T, self.n_heads, self.d_head)
        V = V.reshape(B, T, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scores = torch.matmul(
            Q, K.transpose(-1, -2))/(
                self.d_head**0.5) #[B,n_heads,T,T]
        mask = torch.tril(torch.ones(
            T,T, device=x.device).view(1,1,T,T))
        scores = scores.masked_fill(
            mask == 0,float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)








