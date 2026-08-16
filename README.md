
# Mini GPT from Scratch

A **GPT-style language model built from scratch in PyTorch**, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The project implements the core components of a decoder-only Transformer and includes training, evaluation, controlled architecture experiments, and text generation.

> **Goal:** Build and understand a complete small-scale language model pipeline rather than fine-tune an existing LLM.

## Highlights

* **~27M parameter** GPT-style decoder-only Transformer
* BPE tokenizer with **8,000-token vocabulary**
* Causal multi-head self-attention implemented from scratch
* Pre-LayerNorm Transformer architecture
* **125K training steps** on TinyStories
* Validation **loss: 1.3663**
* Validation **perplexity: 3.9210**
* Controlled experiments on **depth and attention heads**
* Text generation from short prompts
* Training and evaluation performed with PyTorch on an NVIDIA GPU

---

## Architecture

```text
Input Text
    ↓
BPE Tokenizer
    ↓
Token IDs
    ↓
Token + Positional Embeddings
    ↓
6 × Transformer Blocks
    ├── Pre-LayerNorm
    ├── Causal Multi-Head Self-Attention
    ├── Residual Connection
    ├── Pre-LayerNorm
    ├── Feed-Forward Network (GELU)
    └── Residual Connection
    ↓
Final LayerNorm
    ↓
Linear Language Model Head
    ↓
Next-Token Logits
```

### Core implementation

The model implements:

* Token embeddings
* Learned positional embeddings
* Multi-head causal self-attention
* Scaled dot-product attention
* Causal masking
* Feed-forward networks
* GELU activation
* Pre-LayerNorm
* Residual connections
* Next-token prediction

---

## Model Configuration

| Parameter          |                            Value |
| ------------------ | -------------------------------: |
| Architecture       |            GPT-style Transformer |
| Parameters         |                             ~27M |
| Vocabulary         |                            8,000 |
| `d_model`          |                              512 |
| Transformer layers |                                6 |
| Attention heads    |                                8 |
| Head dimension     |                               64 |
| Context length     |                       256 tokens |
| Attention          | Causal multi-head self-attention |
| FFN activation     |                             GELU |
| Normalization      |                    Pre-LayerNorm |
| Optimizer          |                            AdamW |
| Learning rate      |                         3 × 10⁻⁴ |
| Batch size         |                                8 |
| Training steps     |                          125,000 |
| Hardware           |              NVIDIA GPU (Kaggle) |

---

## Dataset

**TinyStories** was used for both training and validation.

| Split      |  Examples |
| ---------- | --------: |
| Training   | 2,119,719 |
| Validation |    21,990 |

A custom **BPE tokenizer** was trained on the training corpus with an 8,000-token vocabulary.

The model is trained using the standard autoregressive objective:

```text
Input:  The cat sat
Target: cat sat down
```

---

## Training

The training pipeline consists of:

```text
TinyStories
    ↓
BPE Tokenizer
    ↓
Token sequences
    ↓
Context windows
    ↓
GPT
    ↓
Cross-Entropy Loss
    ↓
AdamW
    ↓
Model Checkpoint
```

The baseline was trained for **125,000 steps**.

### Baseline training

```text
Final training loss: 1.7137
Training steps:      125,000
```

The model checkpoint contains both model parameters and optimizer state, allowing training to be resumed.

---

## Evaluation

Evaluation uses the TinyStories validation split with gradients disabled.

**Perplexity** is calculated as:

```text
Perplexity = exp(validation loss)
```

Lower perplexity indicates better next-token prediction performance.

### Baseline

| Metric             |     Result |
| ------------------ | ---------: |
| Validation loss    | **1.3663** |
| Perplexity         | **3.9210** |
| Validation batches |        404 |

---

## Architecture Experiments

Controlled experiments were performed by changing one architectural variable while keeping the other primary settings fixed.

| Experiment   | Layers | Heads | Context | `d_model` |  Val. Loss | Perplexity | Steps |
| ------------ | -----: | ----: | ------: | --------: | ---------: | ---------: | ----: |
| **Baseline** |  **6** | **8** | **256** |       512 | **1.3663** | **3.9210** |  125K |
| Exp 2        |      4 |     8 |     256 |       512 |     1.5682 |     4.7978 |   60K |
| Exp 3        |      8 |     8 |     256 |       512 |     1.4600 |     4.3059 |   60K |
| Exp 4        |      6 |     4 |     256 |       512 |     1.5132 |     4.5411 |   60K |
| Exp 5        |      6 |     8 |     512 |       512 |     2.7390 |    15.4721 |   60K |

### Observations

* Reducing depth from **6 → 4 layers** degraded validation performance.
* Increasing depth to **8 layers** improved performance relative to the 4-layer experiment at the same 60K-step budget.
* Reducing attention heads from **8 → 4** also degraded performance.
* The 512-token context experiment showed substantially higher validation loss at 60K steps.

**Important:** The baseline was trained for 125K steps, while the architecture experiments were trained for 60K steps. Therefore, the experiments should be interpreted as **comparisons at a fixed training-step budget**, not definitive evidence that one architecture is intrinsically superior.

---

## Text Generation

The trained ~27M-parameter model generates coherent TinyStories-style narratives from short prompts.

### Example 1

**Prompt:** `Once upon a time`

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's mom asked her to help her with the laundry...

### Example 2

**Prompt:** `A little boy went to the park`

> A little boy went to the park with his mom. He saw a big tree with lots of leaves. He wanted to climb it. His mom said, "Be careful, don't go too high."...

### Example 3

**Prompt:** `Lily found a strange box`

> Lily found a strange box in the attic. She opened it and found a big, hairy spider inside. Lily was scared and didn't know what to do...

These examples demonstrate that the model learned **TinyStories-style vocabulary, syntax, and short-range narrative structure**.

---

## Project Structure

```text
mini-gpt-from-scratch/
│
├── src/
│   ├── model.py          # GPT architecture
│   ├── dataset.py        # Dataset and tokenization pipeline
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Loss and perplexity evaluation
│   └── generate.py       # Text generation
│
├── tokenizer/
│   └── tinystories.json  # Trained BPE tokenizer
│
├── results/
│   ├── experiments.csv
│   └── generated_samples.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Run Locally

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd mini-gpt-from-scratch
```

Create an environment:

```bash
python -m venv gptvenv
source gptvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python src/train.py
```

Evaluate:

```bash
python src/evaluate.py
```

Generate text:

```bash
python src/generate.py
```

---

## Technologies

* Python
* PyTorch
* Hugging Face Datasets
* Hugging Face Tokenizers
* CUDA
* Kaggle GPU

---

## What This Project Demonstrates

This project demonstrates hands-on understanding of the **end-to-end LLM training pipeline**:

**tokenization → dataset preparation → Transformer architecture → causal attention → training → checkpointing → evaluation → experimentation → generation**

