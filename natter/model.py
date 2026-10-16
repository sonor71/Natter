from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight


def apply_rope(x, cos, sin):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((-x1, x0), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, rope_theta):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError('dim / n_heads должен быть чётным целым числом для RoPE')
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        inv_freq = 1 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(t, device=x.device, dtype=torch.float32)
        theta = torch.outer(positions, self.inv_freq.to(x.device))
        cos = torch.repeat_interleave(theta.cos(), 2, dim=-1)[None, None].to(q.dtype)
        sin = torch.repeat_interleave(theta.sin(), 2, dim=-1)[None, None].to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(math.ceil((8 * dim / 3) / 128) * 128)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dropout, rope_theta)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class NatterTransformer(nn.Module):
    def __init__(self, vocab_size, context_size, dim, n_layers, n_heads, dropout=0.0, rope_theta=10000.0):
        super().__init__()
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, dropout, rope_theta) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.shape[1] > self.context_size:
            raise ValueError(f'Слишком длинный контекст: {idx.shape[1]} > {self.context_size}')
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, prefix, max_new_tokens=120, temperature=0.8, top_k=40, eos_id=None):
        self.eval()
        result = prefix
        for _ in range(max_new_tokens):
            logits, _ = self(result[:, -self.context_size:])
            logits = logits[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.shape[-1])
                    threshold = torch.topk(logits, k).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, -float('inf'))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            result = torch.cat((result, nxt), dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return result
