import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TinyStoriesDataset
from model import GPT

import os

os.makedirs("/content/drive/MyDrive/mini-gpt/checkpoints", exist_ok=True)

import model

# Configuration
batch_size = 8
context_length = 256
learning_rate = 3e-4
epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

# Dataset
dataset = TinyStoriesDataset(
    split="train",
    context_length=context_length
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)
print("Dataset size:", len(dataset))
print("Batches:", len(dataloader))


# Model
model = GPT(
    vocab_size=8000,
    d_model=512,
    context_length=256,
    n_layers=6,
    n_heads=8
).to(device)


print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# Training loop
for epoch in range(epochs):

    for step, (x, y) in enumerate(dataloader):

        x = x.to(device)
        y = y.to(device)

        # Forward
        logits = model(x)

        # Loss
        loss = criterion(
            logits.view(-1, 8000),
            y.view(-1)
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 1000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item()
            }, f"checkpoint_{step}.pt")

        if step % 100 == 0:
            print(
                f"epoch={epoch} "
                f"step={step} "
                f"loss={loss.item():.4f}"
            )

