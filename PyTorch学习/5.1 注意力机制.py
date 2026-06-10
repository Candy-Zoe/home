# PyTorch注意力机制学习
# 主要内容：自注意力机制、多头注意力

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== 自注意力机制 ===")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

embed_dim = 64
seq_len = 10
batch_size = 2

x = torch.randn(batch_size, seq_len, embed_dim)
attention = SelfAttention(embed_dim)
output, weights = attention(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")

print("\n=== 多头注意力 ===")

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

multi_head_attn = MultiHeadAttention(embed_dim, num_heads=4)
output, weights = multi_head_attn(x)
print(f"多头注意力输出形状: {output.shape}")
print(f"多头注意力权重形状: {weights.shape}")