from tinygrad import Tensor, nn, dtypes
from tinygrad import TinyJit
from dataclasses import dataclass
from pathlib import Path
from tqdm import trange, tqdm
from tinygrad.helpers import Timing
from os import environ

from tinygpt.tokenizer import Tokenizer
from tinygpt.model import GPTConfig, GPTLanguageModel


@dataclass
class TrainConfig:
    input_file = "data/input_rick_morty.txt"
    model_name = "gpt2-small"
    lr: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    batch_size: int = 64
    dtype = dtypes.float32
    small_model: bool = environ.get("SMALL", "false").lower() == "true"

    @property
    def dataset_name(self):
        return Path(self.input_file).stem.replace("input_", "")

    def model_filename(self, model_cfg: GPTConfig) -> str:
        return (
            f"{self.model_name}-{self.dataset_name}-{model_cfg.n_embed}e"
            f"-{model_cfg.n_layers}l-{model_cfg.n_heads}h-{model_cfg.block_size}b"
            f"-{self.batch_size}bs-{self.dtype.name}.safetensors"
        )


tc = TrainConfig()


def train():
    tok = Tokenizer.from_file(tc.input_file)
    with open(tc.input_file, "r", encoding="utf-8") as _f:
        text = _f.read()

    dtypes.default_float = tc.dtype
    print(f"using dtype={tc.dtype}")

    model_cfg = (
        GPTConfig.small(vocab_size=tok.vocab_size)
        if tc.small_model
        else GPTConfig(vocab_size=tok.vocab_size)
    )
    m = GPTLanguageModel(model_cfg)

    params = nn.state.get_parameters(m)
    print(sum(p.numel() for p in params) / 1e6, "M parameters")

    optimizer = nn.optim.AdamW(params, lr=tc.lr)

    data = Tensor(tok.encode(text), dtype=dtypes.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(is_training=True):
        d = train_data if is_training else val_data
        ix = Tensor.randint((tc.batch_size, 1), high=len(d) - model_cfg.block_size)
        offsets = Tensor.arange(model_cfg.block_size).reshape(1, model_cfg.block_size)
        idx = ix + offsets
        x = d[idx]
        y = d[idx + 1]
        return x, y

    @TinyJit
    def eval_step(xb, yb):
        _, loss = m(xb, yb)
        return loss.realize()

    def estimate_loss():
        out = {}
        with Tensor.train(False):
            for split in ["train", "val"]:
                losses = []
                for _ in range(tc.eval_iters):
                    X, Y = get_batch(is_training=(split == "train"))
                    loss = eval_step(X, Y)
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
        return out

    @TinyJit
    def step(xb, yb):
        _, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.realize()

    with Timing("total training: "):
        with Tensor.train():
            xb, yb = get_batch(is_training=True)
            with Timing("  warmup step: "):
                loss = step(xb, yb)

            pbar = trange(tc.max_iters, desc="training")
            for iter in pbar:
                if iter % tc.eval_interval == 0 or iter == tc.max_iters - 1:
                    losses = estimate_loss()
                    tqdm.write(
                        f"step {iter:>5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )

                xb, yb = get_batch(is_training=True)
                loss = step(xb, yb)

    state_dict = nn.state.get_state_dict(m)
    model_file = tc.model_filename(model_cfg)
    nn.state.safe_save(state_dict, model_file)
    print(f"model saved to {model_file}")
