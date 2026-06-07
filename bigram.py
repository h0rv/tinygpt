import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import httpx

    return (httpx,)


@app.cell
def _(httpx):
    from pathlib import Path

    input_path = Path("input.txt")

    if not input_path.is_file():
        res = httpx.get(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
        with open(input_path, "w", encoding="utf-8") as _f:
            _f.write(res.text)
    return (input_path,)


@app.cell
def _(input_path):
    with open(input_path, "r", encoding="utf-8") as _f:
        text = _f.read()
    len(text)
    return (text,)


@app.cell
def _(text):
    print(text[:500])
    return


@app.cell
def _(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("".join(chars))
    print(vocab_size)
    return chars, vocab_size


@app.cell
def _():
    from tinygrad import Tensor, dtypes, nn
    import numpy as np

    return Tensor, dtypes, nn


@app.cell
def _(chars):
    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for i, ch in enumerate(chars)}


    def encode(s: str) -> list[int]:
        return [char2int[ch] for ch in s]


    def decode(ints: int | list[int]) -> str:
        if isinstance(ints, int):
            ints = [ints]
        return "".join([int2char[i] for i in ints])


    print(encode("hello, world!"))
    print(decode(encode("hello, world!")))
    return decode, encode


@app.cell
def _(Tensor, dtypes, encode, text):
    data = Tensor(encode(text), dtype=dtypes.long)
    print(data.shape, data.dtype)
    print(data[:100].numpy())
    return (data,)


@app.cell
def _(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


@app.cell
def _(Tensor, decode, train_data, val_data):
    Tensor.manual_seed(1337)

    batch_size = 4  # number of independent sequences to process in parallel
    block_size = 8  # maximum context length for predictions


    def get_batch(is_training=True):
        data = train_data if is_training else val_data
        # ix = Tensor.randint((batch_size,), high=len(data) - block_size).tolist()
        # x = Tensor.stack([data[i : i + block_size] for i in ix])
        # y = Tensor.stack([data[i + 1 : i + block_size + 1] for i in ix])
        # fully lazy: sample starts on-device and gather, no .tolist() sync
        ix = Tensor.randint((batch_size, 1), high=len(data) - block_size)
        offsets = Tensor.arange(block_size).reshape(1, block_size)
        idx = ix + offsets  # (batch_size, block_size)
        x = data[idx]  # tensor gather -> (batch_size, block_size)
        y = data[idx + 1]
        return x, y


    xb, yb = get_batch(is_training=True)
    print("inputs:")
    print(f"  {xb.shape}")
    print(f"  {xb.numpy()}")
    print("targets:")
    print(f"  {yb.shape}")
    print(f"  {yb.numpy()}")

    print("-----")

    for b in range(batch_size):  # batch dimension
        for t in range(block_size):  # time dimension
            context = xb[b, : t + 1]
            target = yb[b, t]
            print(f"{context.shape=}, {target.shape=}")
            print(
                f"when input is {context.tolist()} ({decode(context.tolist())}) the target is {target.tolist()} ({decode(target.tolist())})"
            )
    return get_batch, xb, yb


@app.cell
def _(Tensor, nn, vocab_size, xb, yb):
    Tensor.manual_seed(1337)


    class BigramLanguageModel:
        def __init__(self, vocab_size):
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def __call__(self, idx, targets=None):
            # idx and targets are both (B, T) tensors of integers
            logits = self.token_embedding_table(idx)  # (B,T,C)

            loss = None
            if targets is not None:
                B, T, C = logits.shape
                logits = logits.reshape(B * T, C)
                targets = targets.reshape(B * T)
                loss = logits.cross_entropy(targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # get the predictions
                logits, _ = self(idx)
                # focus only on the last time step
                logits = logits[:, -1, :]  # becomes (B, C)
                # apply softmax to get probabilities
                probs = logits.softmax(axis=-1)  # (B, C)
                # sample from the distribution
                idx_next = probs.multinomial(num_samples=1)  # (B, 1)
                # append sampled index to the running sequence
                idx = idx.cat(idx_next, dim=1).realize()  # (B, T+1)
            return idx


    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss.tolist())
    return (m,)


@app.cell
def _(Tensor, decode, dtypes, m):
    def gen(max_new_tokens=100):
        idx = Tensor.zeros(1, 1, dtype=dtypes.long)
        print(decode(m.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()))


    gen()
    return (gen,)


@app.cell
def _(m, nn):
    optimizer = nn.optim.AdamW(nn.state.get_parameters(m), lr=1e-3)
    return (optimizer,)


@app.cell
def _(Tensor, get_batch, m, optimizer):
    from tinygrad import TinyJit

    _batch_size = 32


    @TinyJit
    def step(xb, yb):
        _, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()


    Tensor.training = True
    for _ in range(100000):
        _loss = step(*get_batch(is_training=True))


    Tensor.training = False
    print(_loss.item())
    return


@app.cell
def _(gen):
    gen(max_new_tokens=1000)
    return


if __name__ == "__main__":
    app.run()
