from tinygrad import Tensor, nn, dtypes
from tinygrad import TinyJit
from dataclasses import dataclass
import httpx
from pathlib import Path

Tensor.manual_seed(1337)


@dataclass
class HyperParams:
    batch_size = 64  # number of independent sequences to process in parallel
    block_size = 256  # maximum context length for predictions
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    learning_rate = 3e-4
    num_embed = 384
    num_heads = 6
    num_layer = 6
    dropout = 0.2
    head_size = num_embed // num_heads


url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
input_path = Path("input.txt")

if not input_path.is_file():
    res = httpx.get(url)
    with open(input_path, "w", encoding="utf-8") as _f:
        _f.write(res.text)

with open(input_path, "r", encoding="utf-8") as _f:
    text = _f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2int = {ch: i for i, ch in enumerate(chars)}
int2char = {i: ch for i, ch in enumerate(chars)}


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
    ix = Tensor.randint(
        (HyperParams.batch_size, 1), high=len(data) - HyperParams.block_size
    )
    offsets = Tensor.arange(HyperParams.block_size).reshape(1, HyperParams.block_size)
    idx = ix + offsets  # (batch_size, block_size)
    x = data[idx]  # tensor gather -> (batch_size, block_size)
    y = data[idx + 1]
    return x, y


def estimate_loss(model, eval_iters: int = HyperParams.eval_iters):
    out = {}
    with Tensor.train(False):  # disable dropout for evaluation
        for split in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                X, Y = get_batch(is_training=(split == "train"))
                _, loss = model(X, Y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    return out


class SelfAttentionHead:
    """
    one head of self-attention
    """

    def __init__(
        self,
        n_embed: int = HyperParams.num_embed,
        head_size: int = HyperParams.head_size,
        block_size: int = HyperParams.block_size,
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
        wei = wei.dropout(HyperParams.dropout)
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
        n_heads: int = HyperParams.num_heads,
        n_embed: int = HyperParams.num_embed,
        head_size: int = HyperParams.head_size,
        block_size: int = HyperParams.block_size,
    ):
        self.heads = [
            SelfAttentionHead(n_embed, head_size, block_size) for _ in range(n_heads)
        ]
        self.proj = nn.Linear(head_size * n_heads, n_embed)

    def __call__(self, x: Tensor):
        out = Tensor.cat(*[h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = out.dropout(HyperParams.dropout)
        return out


class FeedForward:
    """ """

    def __init__(
        self, n_embed: int = HyperParams.num_embed, dropout: float = HyperParams.dropout
    ):
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
        n_embed: int = HyperParams.num_embed,
        head_size: int = HyperParams.head_size,
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
        block_size: int = HyperParams.block_size,
        n_embed: int = HyperParams.num_embed,
        n_head: int = HyperParams.num_heads,
        n_layers: int = HyperParams.num_layer,
    ):
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = [TransformerBlock(n_embed) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

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

    def generate(self, idx, max_new_tokens, block_size: int = HyperParams.block_size):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = logits.softmax(axis=-1)  # (B, C)
            # sample from the distribution
            idx_next = probs.multinomial(num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = idx.cat(idx_next, dim=1).realize()  # (B, T+1)
        return idx


################################################################


def main():
    m = GPTLanguageModel(vocab_size)

    # print the number of parameters in the model
    params = nn.state.get_parameters(m)
    print(sum(p.numel() for p in params) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = nn.optim.AdamW(params, lr=HyperParams.learning_rate)

    @TinyJit
    def step(xb, yb):
        _, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

    for iter in range(HyperParams.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % HyperParams.eval_interval == 0 or iter == HyperParams.max_iters - 1:
            losses = estimate_loss(m)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch(is_training=True)

        # evaluate the loss
        loss = step(xb, yb)

    # generate from the model
    context = Tensor.zeros(1, 1, dtype=dtypes.long)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


if __name__ == "__main__":
    main()
