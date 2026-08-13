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


train: Dataset({features: ['text'], num_rows: 2119719})
validation: Dataset({features: ['text'],num_rows: 21990})


Step: 125000
Loss: 1.713707447052002

Base evaluation: 

## Model Evaluation

| Parameter | Value |
|---|---:|
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

Step: 60000
Loss: 1.7748910188674927

