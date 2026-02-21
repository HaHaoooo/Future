# -*- coding: utf-8 -*-
"""
Transformer 核心：因果自注意力 + RoPE + Pre-Norm

架构对齐 LLaMA / Mistral / Gemma：
  - RMSNorm 替代 LayerNorm（更高效，收敛更快）
  - SwiGLU FFN 替代 GELU FFN（门控激活，语言建模效果更优）
  - RoPE 旋转位置编码（支持长上下文外推）
  - Flash Attention（PyTorch 2.0+ 自动启用）
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_cos_sin(dim: int, seq_len: int, base: float = 10000.0, device=None):
    """RoPE cos/sin：支持任意序列长度，利于长上下文外推。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """LLaMA 风格 RoPE：x1=x[...,:d/2], x2=x[...,d/2:], 返回 [-x2, x1]"""
    d = x.shape[-1]
    return torch.cat([-x[..., d // 2 :], x[..., : d // 2]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
    """
    RoPE 应用于 q, k。q,k: (B, num_heads, T, head_dim)
    返回 (q_rotated, k_rotated)。
    """
    head_dim = q.shape[-1]
    cos, sin = _rope_cos_sin(head_dim, seq_len, device=q.device)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class RMSNorm(nn.Module):
    """RMSNorm (LLaMA/Gemma 风格)：比 LayerNorm 更高效，收敛更快。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN (LLaMA/Mistral/Gemma 标配)：门控激活，语言建模效果优于 GELU FFN。"""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """因果多头自注意力 + RoPE，支持 Flash Attention。"""

    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * num_heads * self.head_dim)
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k, T)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(B, T, -1)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 块：RMSNorm → Attn → residual, RMSNorm → SwiGLU → residual。"""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout=dropout)
        self.ln2 = RMSNorm(dim)
        # SwiGLU 有 3 个矩阵，按 2/3 缩放保持参数量一致
        swiglu_dim = int(dim * ffn_mult * 2 / 3)
        self.ffn = SwiGLU(dim, swiglu_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class XiaolaiTransformer(nn.Module):
    """
    小来 Transformer 编码器：多层 causal decoder，输出最后一位置的隐态。
    支持任意序列长度（受 context_max_len 裁剪）。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        sensory_dim: int,
        emotion_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 8192,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = hidden_dim + sensory_dim + emotion_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.sensory_proj = nn.Linear(sensory_dim, sensory_dim)
        self.emotion_proj = nn.Linear(emotion_dim, emotion_dim)
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_mult=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(hidden_dim)
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(
        self,
        token_ids: torch.Tensor,
        sensory: torch.Tensor,
        emotion: torch.Tensor,
    ) -> torch.Tensor:
        """
        token_ids: (B, T), sensory: (B, S), emotion: (B, E)
        返回 (B, hidden_dim) 最后一位置隐态。
        """
        B, T = token_ids.shape
        if T > self.max_seq_len:
            token_ids = token_ids[:, -self.max_seq_len :]
            T = self.max_seq_len
        emb = self.embed(token_ids)
        s = self.sensory_proj(sensory)
        e = self.emotion_proj(emotion)
        s_expand = s.unsqueeze(1).expand(-1, T, -1)
        e_expand = e.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([emb, s_expand, e_expand], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x[:, -1, :]
