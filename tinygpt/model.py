from tinygrad import Tensor, nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_embed: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.2

    @property
    def head_size(self):
        return self.n_embed // self.n_heads

    @classmethod
    def small(cls, vocab_size: int) -> "GPTConfig":
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=64,
            n_embed=64,
            n_heads=4,
            n_layers=3,
        )


class SelfAttentionHead:
    def __init__(self, cfg: GPTConfig):
        self.key = nn.Linear(cfg.n_embed, cfg.head_size, bias=False)
        self.query = nn.Linear(cfg.n_embed, cfg.head_size, bias=False)
        self.value = nn.Linear(cfg.n_embed, cfg.head_size, bias=False)
        self.tril = Tensor.ones(cfg.block_size, cfg.block_size).is_param_(False).tril()
        self.dropout = cfg.dropout

    def __call__(self, x: Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei.softmax(axis=-1)
        wei = wei.dropout(self.dropout)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention:
    def __init__(self, cfg: GPTConfig):
        self.heads = [SelfAttentionHead(cfg) for _ in range(cfg.n_heads)]
        self.proj = nn.Linear(cfg.head_size * cfg.n_heads, cfg.n_embed)
        self.dropout = cfg.dropout

    def __call__(self, x: Tensor):
        out = Tensor.cat(*[h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = out.dropout(self.dropout)
        return out


class FeedForward:
    def __init__(self, cfg: GPTConfig):
        self.fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)
        self.proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed)
        self.layers = [
            self.fc,
            Tensor.relu,
            self.proj,
            lambda x: x.dropout(cfg.dropout),
        ]

    def __call__(self, x: Tensor):
        return x.sequential(self.layers)


class TransformerBlock:
    def __init__(self, cfg: GPTConfig):
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def __call__(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel:
    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

        self._init_weights(cfg.n_layers)

    def _init_weights(self, n_layer: int):
        def normal_(t: Tensor, std: float = 0.02):
            t.assign(Tensor.normal(*t.shape, mean=0.0, std=std))

        def walk(obj):
            children = obj if isinstance(obj, (list, tuple)) else vars(obj).values()
            for c in children:
                if isinstance(c, (nn.Linear, nn.Embedding)):
                    normal_(c.weight)
                    if getattr(c, "bias", None) is not None:
                        c.bias.assign(Tensor.zeros(*c.bias.shape))
                elif isinstance(c, (list, tuple)) or (
                    hasattr(c, "__dict__") and not isinstance(c, Tensor)
                ):
                    walk(c)

        walk(self)

        scale = 0.02 * (2 * n_layer) ** -0.5
        for block in self.blocks:
            normal_(block.sa.proj.weight, std=scale)
            normal_(block.ffwd.proj.weight, std=scale)

    def __call__(self, idx: Tensor, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(Tensor.arange(T))
        x = tok_emb + pos_emb
        x = x.sequential(self.blocks)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = logits.cross_entropy(targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = logits.softmax(axis=-1)
            idx_next = probs.multinomial(num_samples=1)
            idx = idx.cat(idx_next, dim=1).realize()
        return idx
