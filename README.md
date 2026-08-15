50m-tinystories/
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── evaluate.py
│   └── train.py
├── tokenizer/
│   └── tinystories.json
├── checkpoint_latest.pt
├── requirements.txt
├── README.md
└── .gitignore


✅ Baseline — 6 layers, 8 heads, context 256
✅ Exp 2 — 4 layers
✅ Exp 3 — 8 layers
✅ Exp 4 — 6 layers, 4 heads
✅ Exp 5 — 6 layers, 8 heads, context 512


| Experiment | Layers | Heads | Context | d-model  | Validation Loss | Perplexity | Training steps |
| ---------- | -----: | ----: | ------: | ------- | --------------- | ---------- | -------------- |
| Baseline   |      6 |     8 |     256 |   512   |     1.3663      |   3.9210    |  125,000      |
| Exp 2      |      4 |     8 |     256 |   512   |     1.5682      |   4.7978    |   60,000      |   
| Exp 3      |      8 |     8 |     256 |   512   |     1.4600      |   4.3059    |   60,000      |  
| Exp 4      |      6 |     4 |     256 |   512   |     1.5132      |   4.5411    |   60,000      |  
| Exp 5      |      6 |     8 |     512 |   512   |     2.7390      |   15.4721   |   60,000      |  



train: Dataset({features: ['text'], num_rows: 2119719})
validation: Dataset({features: ['text'],num_rows: 21990})


# 6-layer Baseline 
training:
Step: 125000
Loss: 1.713707447052002

Base evaluation: 

## Model Evaluation

| Parameter | Value |
| Model architecture | GPT-style Transformer |
| Parameters | ~27M |
| Training dataset | TinyStories |
| Evaluation dataset | TinyStories validation |
| Vocabulary size | 8,000 |
| Context length | 256 tokens |
| Embedding dimension (`d_model`) | 512 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Head dimension | 64 |
| Attention type | Causal multi-head self-attention |
| Feed-forward activation | GELU |
| Normalization | Pre-LayerNorm |
| Training steps | 125,000 |
| Batch size | 8 |
| Learning rate | 3 × 10⁻⁴ |
| Optimizer | AdamW |
| Validation batches | 404 |
| Validation loss | **1.3663** |
| Perplexity | **3.9210** |
| Evaluation mode | `torch.no_grad()` |
| Hardware | NVIDIA GPU (Kaggle) |

# 4-layer 

Exp 2
training:
Step: 60000
Loss: 1.774

layers: 4,
Heads: 8,
Context: 256

Evaluation:

Validation Loss: 1.5682
Perplexity: 4.7978


# 8 layer 

Exp3
Layers: 8
Heads: 8
Context Length: 256
Parameters: ~34
Step: 60000
Loss: 1.6350
Checkpoint: checkpoint_latest_8layer.pt

Evaluation: 

Validation Loss: 1.4600
Perplexity: 4.3059

# 4heads

Exp 4
Layers: 6
Heads: 4
Context: 256
Parameters: ~27M
Steps: 60,000
Training loss: 1.7002
Checkpoint: checkpoint_latest_4head.pt

Evaluation:

Validation Loss: 1.5132
Perplexity: 4.5411



# context=512

Exp 5
Layers: 6
Heads: 8
Context: 512
Parameters: ~27M
Training Steps: 60,000
Training loss: 1.6418
Checkpoint: checkpoint_latest_context512.pt

Evaluation:

Validation Loss: 2.7390
Perplexity: 15.4721







## Text Generation

The trained 27M-parameter GPT can generate coherent TinyStories-style text
from short prompts.

### Example 1

**Prompt:** `Once upon a time`

> Once upon a time, there was a little girl named Lily. She loved to play
> with her toys and her favorite toy was a teddy bear. One day, Lily's mom
> asked her to help her with the laundry...

### Example 2

**Prompt:** `A little boy went to the park`

> A little boy went to the park with his mom. He saw a big tree with lots
> of leaves. He wanted to climb it. His mom said, "Be careful, don't go
> too high." The little boy was very careful...

### Example 3

**Prompt:** `Lily found a strange box`

> Lily found a strange box in the attic. She opened it and found a big,
> hairy spider inside. Lily was scared and didn't know what to do...