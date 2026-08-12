import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TinyStoriesDataset
from model import GPT


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
    n_layers=4,
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
'''
checkpoint = torch.load(
    "/kaggle/working/checkpoint_62000.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_step = checkpoint["step"] + 1

print("Resuming from step:", start_step)
'''


for epoch in range(epochs):

    for step, (x, y) in enumerate(dataloader):
        if step < start_step:
            continue

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

        if step % 5000 == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item()
            }, "/kaggle/working/checkpoint_latest_4layer.pt")


        if step % 100 == 0:
            print(
                f"epoch={epoch} "
                f"step={step} "
                f"loss={loss.item():.4f}"
            )

