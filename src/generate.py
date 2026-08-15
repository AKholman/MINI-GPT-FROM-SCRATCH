import torch

from model import GPT
from tokenizers import Tokenizer


# Configuration
vocab_size = 8000
d_model = 512
context_length = 256
n_layers = 6
n_heads = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# Load tokenizer
tokenizer = Tokenizer.from_file(
    "tokenizer/tinystories.json"
)


# Create model
model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    context_length=context_length,
    n_layers=n_layers,
    n_heads=n_heads
).to(device)


# Load checkpoint
checkpoint = torch.load(
    "checkpoint_latest.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Prompt
prompt = "his car was very"

tokens = tokenizer.encode(prompt).ids

x = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device
)


# Generate
max_new_tokens = 100

with torch.no_grad():

    for _ in range(max_new_tokens):

        # Keep only the latest context
        x_context = x[:, -context_length:]

        logits = model(x_context)

        # Last token prediction
        logits = logits[:, -1, :]

        # Greedy decoding
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )


# Decode
generated_tokens = x[0].tolist()

text = tokenizer.decode(generated_tokens)

print("\nGenerated text:\n")
print(text)