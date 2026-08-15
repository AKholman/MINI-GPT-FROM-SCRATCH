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
🔄 Exp 6 — 6 layers, 8 heads, context 512


| Experiment | Layers | Heads | Context | Status |
| ---------- | -----: | ----: | ------: | ------ |
| Baseline   |      6 |     8 |     256 | ✅      |
| Exp 2      |      4 |     8 |     256 | ✅      |
| Exp 3      |      8 |     8 |     256 | ✅      |
| Exp 4      |      6 |     4 |     256 | ✅      |
| Exp 5      |      6 |     8 |     128 | ❌ Skip |
| Exp 6      |      6 |     8 |     512 | 🔄     |





train: Dataset({features: ['text'], num_rows: 2119719})
validation: Dataset({features: ['text'],num_rows: 21990})


# 6-layer 
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


# 8 layer

Dataset size: 448932815
Batches: 56116602
Parameters: 33551168

Step: 60000
Loss: 1.63507878780365

# 4heads

Exp 4
Layers: 6
Heads: 4
Context: 256
Parameters: 27,246,400
Steps: 60,000
Training loss: 1.7002723217010498
Checkpoint: checkpoint_latest_4head.pt