from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):

    def __init__(self, split="train", context_length=256):

        self.context_length = context_length

        # Load dataset
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train[:10]"
        )

        # Load our tokenizer
        self.tokenizer = Tokenizer.from_file(
            "tokenizer/tinystories.json"
        )

        # Tokenize all stories
        self.tokens = []

        for example in dataset:
            ids = self.tokenizer.encode(
                example["text"]
            ).ids

            self.tokens.extend(ids)

        self.tokens = torch.tensor(
            self.tokens,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):

        x = self.tokens[
            idx : idx + self.context_length
        ]

        y = self.tokens[
            idx + 1 : idx + self.context_length + 1
        ]

        return x, y