import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TinyStoriesDataset
from model import GPT


# Configuration
batch_size = 8
context_length = 256
vocab_size = 8000

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# Validation dataset
dataset = TinyStoriesDataset(
    split="validation",
    context_length=context_length
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)


# Model
model = GPT(
    vocab_size=vocab_size,
    d_model=512,
    context_length=context_length,
    n_layers=6,
    n_heads=4
).to(device)


# Load checkpoint
checkpoint = torch.load(
    "/kaggle/working/checkpoint_latest_4head.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Checkpoint step:", checkpoint["step"])


# Evaluation
criterion = nn.CrossEntropyLoss()

model.eval()

total_loss = 0.0
total_batches = 0

print("Starting evaluation...")
print("Total validation batches:", len(dataloader))

with torch.no_grad():

    for step, (x, y) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        total_loss += loss.item()
        total_batches += 1

        # Progress
        if total_batches % 100 == 0:
            print(
                f"Evaluated batches: "
                f"{total_batches}/{len(dataloader)} "
                f"| loss={loss.item():.4f}"
            )


validation_loss = total_loss / total_batches
perplexity = math.exp(validation_loss)

print()
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Perplexity:      {perplexity:.4f}")