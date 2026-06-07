import httpx
from pathlib import Path


def fetch_shakespeare():
    out = Path("input.txt")
    if out.exists():
        print(f"{out} already exists, skipping")
        return
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    res = httpx.get(url)
    out.write_text(res.text, encoding="utf-8")
    chars = sorted(set(res.text))
    print(f"shakespeare: {len(res.text)} chars, vocab {len(chars)}")


def fetch_rick_morty():
    out = Path("input_rick_morty.txt")
    if out.exists():
        print(f"{out} already exists, skipping")
        return
    from datasets import load_dataset

    ds = load_dataset("Prarabdha/Rick_and_Morty_Transcript", split="train")
    lines = []
    for row in ds:
        speaker = (row["speaker"] or "").strip()
        dialogue = (row["dialouge"] or "").strip()
        if speaker and dialogue:
            lines.append(f"{speaker}:\n{dialogue}")
    text = "\n\n".join(lines)
    out.write_text(text, encoding="utf-8")
    chars = sorted(set(text))
    print(f"rick & morty: {len(lines)} lines, vocab {len(chars)}")


if __name__ == "__main__":
    fetch_shakespeare()
    fetch_rick_morty()
