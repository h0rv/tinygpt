from tinygrad import Tensor, nn, dtypes
from tinygrad import TinyJit
from dataclasses import dataclass
import httpx
from pathlib import Path
from tqdm import trange, tqdm
from tinygrad.helpers import Timing

Tensor.manual_seed(1337)


@dataclass
class Config:
    input_file = "input_rick_morty.txt"
    model_name = "gpt2-small"
    output_file = "output.txt"

    @property
    def dataset_name(self):
        return Path(self.input_file).stem.replace("input_", "")


@dataclass
class HyperParams:
    # small test config — trains in ~30s for quick iteration
    # batch_size = 16
    # block_size = 64
    # max_iters = 200
    # eval_interval = 50
    # eval_iters = 20
    # learning_rate = 3e-4
    # num_embed = 64
    # num_heads = 4
    # num_layer = 3
    # dropout = 0.2
    # head_size = num_embed // num_heads
    # dtype = dtypes.float32

    # real config — overnight on M5 Pro
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 3e-4
    num_embed = 384
    num_heads = 6
    num_layer = 6
    dropout = 0.2
    head_size = num_embed // num_heads
    dtype = dtypes.float32


cfg = Config()
hp = HyperParams()


def load_vocab(path: str | Path) -> tuple[dict[str, int], dict[int, str], int]:
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for i, ch in enumerate(chars)}
    return char2int, int2char, len(chars)


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    input_path = Path(cfg.input_file)

    if not input_path.is_file():
        if cfg.input_file == "input.txt":
            res = httpx.get(url)
            input_path.write_text(res.text, encoding="utf-8")
        else:
            raise FileNotFoundError(
                f"{cfg.input_file} not found — run fetch_rick_morty.py first"
            )

    char2int, int2char, vocab_size = load_vocab(cfg.input_file)
    with open(cfg.input_file, "r", encoding="utf-8") as _f:
        text = _f.read()

    def encode(s: str) -> list[int]:
        return [char2int[ch] for ch in s]

    def decode(ints: int | list[int]) -> str:
        if isinstance(ints, int):
            ints = [ints]
        return "".join([int2char[i] for i in ints])

    data = Tensor(encode(text), dtype=dtypes.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(is_training=True):
        data = train_data if is_training else val_data
        # ix = Tensor.randint((batch_size,), high=len(data) - block_size).tolist()
        # x = Tensor.stack([data[i : i + block_size] for i in ix])
        # y = Tensor.stack([data[i + 1 : i + block_size + 1] for i in ix])
        # fully lazy: sample starts on-device and gather, no .tolist() sync
        ix = Tensor.randint((hp.batch_size, 1), high=len(data) - hp.block_size)
        offsets = Tensor.arange(hp.block_size).reshape(1, hp.block_size)
        idx = ix + offsets  # (batch_size, block_size)
        x = data[idx]  # tensor gather -> (batch_size, block_size)
        y = data[idx + 1]
        return x, y

    @TinyJit
    def eval_step(xb, yb):
        _, loss = m(xb, yb)
        return loss.realize()

    def estimate_loss(eval_iters: int = hp.eval_iters):
        out = {}
        with Tensor.train(False):  # disable dropout for evaluation
            for split in ["train", "val"]:
                losses = []
                for _ in range(eval_iters):
                    X, Y = get_batch(is_training=(split == "train"))
                    loss = eval_step(X, Y)
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
        return out


class SelfAttentionHead:
    """
    one head of self-attention
    """

    def __init__(
        self,
        n_embed: int = hp.num_embed,
        head_size: int = hp.head_size,
        block_size: int = hp.block_size,
    ):
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.tril = Tensor.ones(block_size, block_size).is_param_(False).tril()

    def __call__(self, x: Tensor):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = wei.softmax(axis=-1)  # (B,T,T)
        wei = wei.dropout(hp.dropout)
        # perform the weighted aggregations of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


class MultiHeadAttention:
    """
    multiple heads of attention in parallel
    """

    def __init__(
        self,
        n_heads: int = hp.num_heads,
        n_embed: int = hp.num_embed,
        head_size: int = hp.head_size,
        block_size: int = hp.block_size,
    ):
        self.heads = [
            SelfAttentionHead(n_embed, head_size, block_size) for _ in range(n_heads)
        ]
        self.proj = nn.Linear(head_size * n_heads, n_embed)

    def __call__(self, x: Tensor):
        out = Tensor.cat(*[h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = out.dropout(hp.dropout)
        return out


class FeedForward:
    """ """

    def __init__(self, n_embed: int = hp.num_embed, dropout: float = hp.dropout):
        self.layers = [
            nn.Linear(n_embed, 4 * n_embed),
            Tensor.relu,
            nn.Linear(4 * n_embed, n_embed),
            lambda x: x.dropout(dropout),
        ]

    def __call__(self, x: Tensor):
        return x.sequential(self.layers)


class TransformerBlock:
    """
    transformer block: communication followed by computation
    """

    def __init__(
        self,
        n_embed: int = hp.num_embed,
        head_size: int = hp.head_size,
    ):
        self.sa = MultiHeadAttention(head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def __call__(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel:
    def __init__(
        self,
        vocab_size: int,
        block_size: int = hp.block_size,
        n_embed: int = hp.num_embed,
        n_head: int = hp.num_heads,
        n_layers: int = hp.num_layer,
    ):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = [TransformerBlock(n_embed) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # init weights
        for name, p in nn.state.get_state_dict(self).items():
            if not p.is_param:  # skip buffers (e.g. the causal mask tril)
                continue
            if name.endswith(".bias"):  # zeros for biases
                p.assign(Tensor.zeros(*p.shape))
            elif p.ndim >= 2:  # N(0,0.02) for Linear/Embedding weights
                p.assign(Tensor.normal(*p.shape, mean=0.0, std=0.02))
            # LayerNorm weight (1D, .weight) is left at default ones

    @property
    def model_name(self):
        return (
            f"{cfg.model_name}-{cfg.dataset_name}-{hp.num_embed}e"
            f"-{hp.num_layer}l-{hp.num_heads}h-{hp.block_size}b"
            f"-{hp.batch_size}bs-{hp.dtype.name}.safetensors"
        )

    def __call__(self, idx: Tensor, targets=None):

        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(Tensor.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = x.sequential(self.blocks)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = logits.cross_entropy(targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size: int = hp.block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = logits.softmax(axis=-1)
            idx_next = probs.multinomial(num_samples=1)
            idx = idx.cat(idx_next, dim=1).realize()
        return idx


################################################################


if __name__ == "__main__":
    # set the default dtype (all params/activations will use this)
    dtypes.default_float = hp.dtype
    print(f"using dtype={hp.dtype}")

    m = GPTLanguageModel(vocab_size)

    # print the number of parameters in the model
    params = nn.state.get_parameters(m)
    print(sum(p.numel() for p in params) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = nn.optim.AdamW(params, lr=hp.learning_rate)

    @TinyJit
    def step(xb, yb):
        _, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

    with Timing("total training: "):
        with Tensor.train():
            # measure one warmup step to JIT compile
            xb, yb = get_batch(is_training=True)
            with Timing("  warmup step: "):
                loss = step(xb, yb)

            pbar = trange(hp.max_iters, desc="training")
            for iter in pbar:
                if iter % hp.eval_interval == 0 or iter == hp.max_iters - 1:
                    losses = estimate_loss()
                    tqdm.write(
                        f"step {iter:>5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )

                xb, yb = get_batch(is_training=True)
                loss = step(xb, yb)

    # generate from the model
    print("--- generation ---")
    seed_idx = int(Tensor.randint(1, high=len(data) - hp.block_size).numpy())
    context = data[seed_idx : seed_idx + hp.block_size].reshape(1, hp.block_size)
    with Timing("  generated 500 tokens in "):
        output = m.generate(context, max_new_tokens=500)[0].tolist()
    generated = decode(output)
    print(generated)
    Path(cfg.output_file).write_text(generated, encoding="utf-8")
    print(f"generation saved to {cfg.output_file}")

    # save the model
    state_dict = nn.state.get_state_dict(m)
    nn.state.safe_save(state_dict, m.model_name)
    print(f"model saved to {m.model_name}")
