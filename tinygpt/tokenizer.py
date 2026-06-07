from pathlib import Path


class Tokenizer:
    def __init__(self, char2int: dict[str, int], int2char: dict[int, str]):
        self.char2int = char2int
        self.int2char = int2char
        self.vocab_size = len(char2int)

    def encode(self, s: str) -> list[int]:
        return [self.char2int[ch] for ch in s]

    def decode(self, ints: list[int]) -> str:
        return "".join([self.int2char[i] for i in ints])

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        char2int = {ch: i for i, ch in enumerate(chars)}
        int2char = {i: ch for i, ch in enumerate(chars)}
        return cls(char2int, int2char)
