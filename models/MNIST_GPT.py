import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

# RoPE
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute Rotary positional embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"x.shape == {x.shape}, freqs.shape == {freqs_cis.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to Q & K tensors right before scaled_dot_product_attention."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MNIST_GPT_tokenizer:
    def __init__(self, num_img_toks, special_toks):
        self.model_in_special_toks = {tok : (num_img_toks + tok_idx) for tok_idx, tok in enumerate(special_toks)}
        self.model_in_vocab_size = num_img_toks + len(self.model_in_special_toks)
        self.model_out_vocab_size = num_img_toks

    def encode_cls(self, inp):
        cls_encoded = [self.model_in_special_toks[cls_tok] for cls_tok in inp]
        return torch.tensor(cls_encoded, dtype=torch.long).reshape((-1, 1))
    
    def encode(self, x, y):
        x = torch.tensor(x, dtype=torch.long)
        y = self.encode_cls(y)
        return torch.cat((y, x), dim=1)


class MultiheadAttention(nn.Module):
    """Multihead attention module"""
    def __init__(self, model_dim, num_heads, dropout=.1):
        super().__init__()
        assert(model_dim % num_heads == 0)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.c_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        q, k ,v = self.qkv(x).split(self.model_dim, dim=-1)
        B, L, D = q.shape

        q = q.view((B, L, self.num_heads, self.head_dim))
        k = k.view((B, L, self.num_heads, self.head_dim))
        v = v.view((B, L, self.num_heads, self.head_dim)).transpose(1,2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        
        res = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None)
            
        res = res.transpose(1,2).contiguous().view((B, L, -1))
        return self.dropout(self.c_proj(res))

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=.1):
        super().__init__()
        self.c_fc = nn.Linear(model_dim, ff_dim)
        self.c_proj = nn.Linear(ff_dim, model_dim)
        self.SiLU = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        x = self.SiLU(self.c_fc(x))
        return self.dropout(self.c_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = MultiheadAttention(model_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, ff_dim, dropout)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, model_in_vocab_size, model_out_vocab_size, num_layers, max_seq_len, model_dim, num_heads, ff_dim, dropout=.1):
        super().__init__()
        # main layers
        self.tok_embed = nn.Embedding(model_in_vocab_size, model_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        # final layers for generation
        self.ln_f = nn.LayerNorm(model_dim)
        self.gen_head = nn.Linear(model_dim, model_out_vocab_size)
        # Rotary posistional embeddings stuff
        self.max_seq_len = max_seq_len
        
        # self.freqs_cis = precompute_freqs_cis(model_dim // num_heads, self.max_seq_len + 2 * len(self.model_in_special_toks))
        
        # TODO: Fix this stange formula (self.max_seq_len + 2 * num_special_toks)
        # TODO: Fix persistent=True -> switch to False and see what happen
        freqs_cis = precompute_freqs_cis(model_dim // num_heads, self.max_seq_len + 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

        # proper weight init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # TODO: assert seqlen < self.max_seq_len
    def forward(self, x):
        bsz, seqlen = x.shape

        h = self.tok_embed(x)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)
        final = self.gen_head(self.ln_f(h))
        return final

    @torch.no_grad()
    def generate(self, x, temperature=1.0):
        max_new_toks = 28 * 28
        t = tqdm(range(max_new_toks), leave=False)
        t.set_description("Generating samples")
        
        for _ in t:
            logits = self(x)
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return x
