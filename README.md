# Grokking Modular Arithmetic : A Mechanistic Interpretation

The core question driving this project is:

> **Why do neural networks suddenly generalize after long periods of memorization, and what internal structural changes make this possible?**

## What Is Grokking?
*Grokking* refers to a training phenomenon where:
1. A neural network fits the training data early
2. Test accuracy remains near chance for a long time
3. Suddenly, test accuracy jumps sharply to near-perfect
This behavior is especially prominent in **algorithmic or algebraic tasks**, such as modular arithmetic.

The dominant hypothesis explored here is:
> **Grokking occurs when the network compresses its internal representations into a low-rank, structured solution that captures the true rule.**

---

## Task: Modular Arithmetic

### Dataset

The task is modular addition:

\[ f(n, m) = (n + m) \bmod p \]

- Inputs: `(n, m)` encoded as **one-hot vectors** of size `2p`
- Outputs: one-hot vector of size `p`
- Dataset size: `p × p`

This task is ideal because:
- It is simple but **highly structured**
- Memorization does not generalize
- The true solution has a **Fourier representation**

---

## Model Architecture

### Model: Mean-Field MLP

```text
(n, m) one-hot → Linear (W1) → Nonlinearity → Linear (W2) → Output
```

Key properties:
- Two-layer MLP
- No bias terms (preserves symmetry)
- Mean-field scaling:
  - Pre-activations scaled by √(input dimension)
  - Output scaled by width

Supported activations:
- Quadratic (default, analytically convenient)
- ReLU
- GELU
This architecture is intentionally simple to make **spectral and rank analysis tractable**.

---

## Training Setup

- Optimizer: AdamW
- Loss: Mean Squared Error (on one-hot outputs)
- Dataset split: random train/test split
- Typical training: 40,000 steps

---

## Core Analysis: Rank and Compression

### Effective Rank

Effective rank measures how many dimensions are *meaningfully used*:

\[ \text{ER} = \exp(H(p_i)) \]

where:
- `p_i` are normalized singular values
- `H` is entropy

Interpretation:
- High ER → diffuse, unstructured representation
- Low ER → compressed, structured solution

Tracked for:
- W1 weights
- W2 weights
- Fourier-transformed weights
- Hidden activations

### Stable Rank

A numerically robust alternative:

\[ \text{Stable Rank} = \frac{||W||_F^2}{||W||_2^2} \]

Used to confirm trends seen in effective rank.

---

## Fourier Analysis

Modular arithmetic has a natural **Fourier decomposition**.

The code:
- Computes FFTs of input-facing weights
- Tracks:
  - Fourier effective rank
  - Inverse Participation Ratio (IPR)
  - Phase alignment of dominant modes

Key idea:

> **Grokking corresponds to the emergence of a small number of dominant Fourier modes.**

---

## Sparsity Analysis

To distinguish *rank compression* from *sparsity*, the following metrics are tracked:

- L0 sparsity (fraction of near-zero weights)
- L1 norm
- Hoyer sparsity
- Activation sparsity

Resulting insight:

> **Grokking is not primarily driven by sparsity—it is driven by low-rank structure.**

---

## Visualization & Figures

The code automatically generates a large suite of figures, including:

### Training Dynamics
- Loss and accuracy
- Weight norms
- Gradient norms
- Effective and stable rank
- Sparsity metrics

### Rank–Grokking Relationship
- Test accuracy vs effective rank
- Rank collapse preceding generalization

### Spectral Analysis
- Singular value spectra (log scale)
- Cumulative explained variance
- Spectral decay rates

### Phase Analysis
- Rank derivatives
- Accuracy derivatives
- Rank–accuracy phase space trajectories

---

## Causal Interventions

Beyond correlation, the code includes **mechanistic tests**:

### Neuron Ablation
- Randomly zero subsets of neurons
- Measure robustness of learned solution

### Weight Perturbation
- Add Gaussian noise to weights
- Measure degradation vs noise scale

### Activation Patching
- Replace activations from clean inputs into corrupted runs
- Tests whether internal representations are **causally sufficient**

These experiments probe *which representations actually matter*.

---

## PCA & Spectral Diagnostics

The framework includes:
- Per-layer singular value spectra
- Explained variance ratios
- Dimensionality required for 95% / 99% variance
- PCA of rank trajectories across layers

These analyses provide a **geometric view of compression**.

---

## Main Experimental Flow

1. Generate modular arithmetic dataset
2. Train Mean-Field MLP
3. Track accuracy, rank, sparsity, Fourier structure
4. Detect compression and grokking phases
5. Perform spectral and PCA analysis
6. Run causal intervention experiments
7. Save all figures

---

## Key Takeaway

> **Grokking corresponds to a sharp transition from a high-rank memorizing solution to a low-rank, Fourier-structured representation that captures the true algorithm.**

This repository provides strong empirical evidence for that claim using both **descriptive metrics** and **causal tests**.

