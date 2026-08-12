from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):

    def __init__(self, split="train", context_length=256):

        self.context_length = context_length

        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split
        )

        self.tokenizer = Tokenizer.from_file(
            "tokenizer/tinystories.json"
        )

        self.tokens = []

        if split == "train":

            # Continuous token stream for training
            for example in dataset:
                ids = self.tokenizer.encode(
                    example["text"]
                ).ids

                self.tokens.extend(ids)

            self.tokens = torch.tensor(
                self.tokens,
                dtype=torch.long
            )

        else:

            # One sequence per validation story
            for example in dataset:

                ids = self.tokenizer.encode(
                    example["text"]
                ).ids

                if len(ids) > context_length:
                    self.tokens.append(
                        torch.tensor(
                            ids[:context_length + 1],
                            dtype=torch.long
                        )
                    )

    def __len__(self):

        if isinstance(self.tokens, torch.Tensor):
            return len(self.tokens) - self.context_length

        return len(self.tokens)

    def __getitem__(self, idx):

        if isinstance(self.tokens, torch.Tensor):

            x = self.tokens[
                idx:idx + self.context_length
            ]

            y = self.tokens[
                idx + 1:idx + self.context_length + 1
            ]

        else:

            sequence = self.tokens[idx]

            x = sequence[:-1]
            y = sequence[1:]

        return x, y