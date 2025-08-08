# Fisher-Stein Scores for Language Model Interpretability

**⚠️ Research Code**: This is experimental software for exploring Fisher-Stein analysis of language models.

A toolkit for computing Fisher Information Matrix-like quantities for understanding how intermediate activations in language models affect next-token predictions.

## What are Fisher-Stein Scores?

Fisher-Stein scores are gradients of log probabilities with respect to intermediate layer activations.

We start with a deep language model $m$ that maps a token $h_k$ and a context $x_{<t}$ to a discrete probability distribution over the next token. We define $m_k$ to be the top layers of $m$ starting from layer $k$. Then we compute:

$$\mathbb{E}_{x_t} [\nabla_{h_k} \log m_k(h_k, x_{<t}) \otimes \nabla_{h_k} \log m_k(h_k, x_{<t})]$$

In very quick PyTorch pseudocode (numerically unstable):

```python
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM('whichever')

def m(h_k, context):
    # h_k are intermediate activations, context is previous tokens
    return softmax(model.upper_layers(h_k, context), dim=-1)

def fisher_stein_score(h_k, context):
    model_jacobian = torch.func.jacrev(m)(h_k, context)  # shape: (vocab_size, hidden_dim)
    # Note: Actual implementation samples tokens and averages across samples
    # for computational tractability rather than computing full jacobian
    return model_jacobian.T @ model_jacobian  # shape: (hidden_dim, hidden_dim)
```

This score bridges two important concepts:
- **Fisher Information Matrix**: Captures second-order dependencies in the model's predictive distribution with respect to its parameters. It is the tangent plane to the statistical manifold defined by the model, and the inverse defines a Riemannian metric.
- **Stein Score**: Similarly defines the tangent plane for the density of the data. 

By computing the expected gradient outer product, we get symmetric PSD matrices whose eigendecomposition reveals interpretable directions in activation space.

## Key Features

- **Split model architecture**: Cleanly separates lower layers (embedding → intermediate) from upper layers (intermediate → logits)
- **Efficient jacobian computation**: Uses `torch.func.jacrev` and `torch.vmap` for batched gradient computation
- **Sampling-based estimation**: Computes Fisher matrices by sampling from the model's predictive distribution rather than summing over the full vocabulary
- **Interpretability focused**: Designed for eigenanalysis to understand activation geometry

## Installation

```bash
git clone https://github.com/yourusername/fisher-stein
cd fisher-stein
pip install -e .
```

## Quick Start

```python
from fisher_stein import LowerLayersModel, UpperLayersModel, calculate_fisher

# Split model at layer 6
lower_model = LowerLayersModel("gpt2", layer_idx=6)
upper_model = UpperLayersModel("gpt2", layer_idx=6)

# Compute Fisher matrices for a batch of sequences
fisher_matrices = calculate_fisher("gpt2", layer_idx=6, batch_tokens)
# Returns: [batch_size, hidden_dim, hidden_dim]

# Eigenanalysis for interpretability
eigenvals, eigenvecs = torch.linalg.eigh(fisher_matrices)
# Project activations onto Fisher eigenvectors to understand their effects
```

## Core Components

### Split Model Architecture

The key insight is splitting the transformer into two parts:

```python
# Lower layers: tokens → intermediate activations
hidden_states, prev_states = lower_model(tokens)

# Upper layers: intermediate activations → next token probabilities  
scores, probs = upper_model.jacobian(final_latents, context)
```

This allows us to:
- Compute gradients with respect to intermediate layers
- Sample next tokens from the model's actual distribution
- Efficiently batch gradient computations

### Fisher Matrix Computation

```python
def fim_expected_gradient_outerproduct(grads, probs):
    """
    Compute Fisher Information Matrix from gradients and probabilities
    
    FIM = E[∇log p(y|x) ∇log p(y|x)^T]
        = ∇^T diag(p) ∇ - (∇^T p)(∇^T p)^T
    """
    weighted_grads = grads * probs.unsqueeze(-1)
    first_term = torch.bmm(grads.transpose(-2, -1), weighted_grads)
    
    weighted_sum = torch.bmm(grads.transpose(-2, -1), probs.unsqueeze(-1))
    second_term = torch.bmm(weighted_sum, weighted_sum.transpose(-2, -1))
    
    return first_term - second_term
```

## Usage Examples

### Basic Fisher Matrix Calculation

```python
import torch
from transformers import AutoTokenizer
from fisher_stein import calculate_fisher

tokenizer = AutoTokenizer.from_pretrained("gpt2")
texts = ["The quick brown fox", "Machine learning is", "In the beginning"]
tokens = tokenizer(texts, return_tensors="pt", padding=True).input_ids

# Calculate Fisher matrices for layer 6
fisher_matrices = calculate_fisher("gpt2", layer_idx=6, tokens)
print(f"Shape: {fisher_matrices.shape}")  # [3, 768, 768]
```

### Large-Scale Analysis with Sampling

```python
# Use the provided script for large-scale analysis
python calculate_fisher_matrices.py \
    --model_name gpt2 \
    --layer_idx 6 \
    --output_dir ./results \
    --batch_size 4 \
    --num_samples 10 \
    --max_seq_len 256
```

## Scaling Considerations

**Current Limitations**: The computational bottleneck is the number of backward passes required: `O(num_samples × hidden_dim)` backward passes through the upper model layers. Forward passes are relatively cheap. Currently tested primarily on GPT-2 scale models.

**Recommended Usage**:
- Start with smaller models (GPT-2, small T5) 
- Focus on specific layers/positions of interest
- Use modest batch sizes and sample counts initially
- Each sample requires one backward pass per vocabulary token sampled

## Theoretical Background

### Connection to Fisher Information

The Fisher Information Matrix measures the curvature of the log-likelihood surface with respect to the _parameters_ $\theta$ of the statistical model:

$$F_{ij} = - \mathbb{E}_{\theta} [ \frac{\partial^2}{\partial \theta_i \partial \theta_j}  (\log p(y|x, \theta))]$$

It gives the tangent plane of the statistical manifold.

The Stein Score is slightly different, defined with respect to the data density $p(x)$:

$$\nabla_x \log p(x) $$ 

Taking a tensor product and an expectation, we similarly get the tangent plane to the data manifold:

$$\mathbb{E}_x [\nabla_x \log p(x) \otimes \nabla_x \log p(x) ]$$ 

Our Fisher-Stein scores compute this with respect to intermediate activations rather than model parameters:

$$\mathbb{E}_h [\nabla_h \log p(y | h) \otimes \nabla_h \log p(y | h)]$$ 

This gives us a symmetric positive semi-definite tangent plane to the model's statistical manifold with respect to the intermediate activations. It tells us precisely how a small nudge to $h_k$ changes the predicted distribution $p(y | h_k)$. Because it is symmetric PSD, we are guaranteed a completely real eigendecomposition.

### Sampling Strategy

Rather than computing gradients for all 50k+ vocabulary tokens, we:

1. Sample next tokens from `p(token | context)` using nucleus sampling
2. Compute gradients only for sampled tokens  
3. Average Fisher matrices across samples

This gives an unbiased estimator while being computationally tractable.

### Numerical Stability

- All matrices are computed in float32 with careful numerical practices
- Eigendecompositions use `torch.linalg.eigh` for symmetric matrices
- Gradient computations use `torch.func` for efficiency and stability

## Research Applications

This toolkit enables research into:

- **Activation geometry**: How do different directions in activation space affect predictions?
- **Layer-wise analysis**: How does Fisher structure change across layers?
- **Attention mechanisms**: What activation patterns drive attention weights?
- **Model editing**: Which activation directions are safe to modify?
- **Mechanistic interpretability**: Understanding transformer computation graphs

## Contributing

**⚠️ Research Status**: This is experimental research code. APIs may change as we explore the most effective computational approaches.

We welcome contributions and feedback! Areas of particular interest:

- **Performance optimization**: Scaling to larger models and datasets
- **Numerical stability**: Edge cases and robustness improvements  
- **Model architectures**: Support for T5, PaLM, LLaMA, etc.
- **Interpretability applications**: Novel uses of Fisher-Stein analysis
- **Validation studies**: Empirical evaluation of the approach

**Before contributing large changes**, please open an issue to discuss the approach.

## License

Apache 2.0 - see LICENSE file for details.