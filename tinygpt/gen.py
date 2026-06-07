#!/usr/bin/env python3
"""Generate text from a trained tinygpt model.

Usage:
  uv run -m tinygpt.gen                           # random generation
  uv run -m tinygpt.gen -p "Morty: "              # from prompt
  uv run -m tinygpt.gen -m model.safetensors -n 200
"""

import argparse
from pathlib import Path

from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import Timing

from tinygpt.tokenizer import Tokenizer
from tinygpt.model import GPTConfig, GPTLanguageModel

DEFAULT_BLOCK_SIZE = 256
DEFAULT_INPUT_FILE = "data/input_rick_morty.txt"


def find_latest_model() -> str:
    models = sorted(
        Path(".").glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not models:
        raise FileNotFoundError("no .safetensors files found")
    return str(models[0])


def parse_model_config(name: str) -> dict:
    parts = name.split("-")
    cfg = {
        "n_embed": 384,
        "n_layers": 6,
        "n_heads": 6,
        "block_size": DEFAULT_BLOCK_SIZE,
    }
    for p in parts:
        if p.endswith("e") and p[:-1].isdigit():
            cfg["n_embed"] = int(p[:-1])
        elif p.endswith("l") and p[:-1].isdigit():
            cfg["n_layers"] = int(p[:-1])
        elif p.endswith("h") and p[:-1].isdigit():
            cfg["n_heads"] = int(p[:-1])
        elif p.endswith("b") and p[:-1].isdigit():
            cfg["block_size"] = int(p[:-1])
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained tinygpt model"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="path to .safetensors model file"
    )
    parser.add_argument("--prompt", "-p", default="", help="input prompt")
    parser.add_argument("--max-new-tokens", "-n", type=int, default=500)
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    args = parser.parse_args()

    model_path = args.model or find_latest_model()
    print(f"loading model: {model_path}")

    tok = Tokenizer.from_file(args.input_file)
    model_params = parse_model_config(Path(model_path).stem)
    print(
        f"vocab={tok.vocab_size} block={model_params['block_size']} embed={model_params['n_embed']} "
        f"layers={model_params['n_layers']} heads={model_params['n_heads']}"
    )

    dtypes.default_float = dtypes.float32
    cfg = GPTConfig(vocab_size=tok.vocab_size, **model_params)
    model = GPTLanguageModel(cfg)
    state_dict = nn.state.safe_load(model_path)
    nn.state.load_state_dict(model, state_dict, strict=False)
    print("model loaded")

    if args.prompt:
        ids = Tensor([tok.encode(args.prompt)], dtype=dtypes.long)
    else:
        ids = Tensor.randint(1, high=tok.vocab_size).reshape(1, 1)

    with Timing("generated in "):
        output = model.generate(ids, max_new_tokens=args.max_new_tokens)[0].tolist()
    print(tok.decode(output))


if __name__ == "__main__":
    main()
