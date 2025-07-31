import torch
from transformer_lm_fim.split_model import LowerLayersModel, UpperLayersModel



def fim_expected_gradient_outerproduct_batch(grads, probs):
    """
    Compute FIM for a batch of gradients and probabilities

    Args:
        grads: [batch_size, vocab_size, hidden_dim]
        probs: [batch_size, vocab_size]

    Returns:
        fim: [batch_size, hidden_dim, hidden_dim]
    """
    # First term: G^T diag(p) G for each batch element
    # grads.transpose(-2, -1): [batch_size, hidden_dim, vocab_size]
    # torch.diag_embed(probs): [batch_size, vocab_size, vocab_size]

    # Compute G^T @ diag(p) @ G efficiently
    weighted_grads = grads * probs.unsqueeze(-1)  # [batch_size, vocab_size, hidden_dim]
    first_term = torch.bmm(grads.transpose(-2, -1), weighted_grads)  # [batch_size, hidden_dim, hidden_dim]

    # Second term: -(G^T p)(G^T p)^T for each batch element
    weighted_sum = torch.bmm(grads.transpose(-2, -1), probs.unsqueeze(-1))  # [batch_size, hidden_dim, 1]
    second_term = torch.bmm(weighted_sum, weighted_sum.transpose(-2, -1))  # [batch_size, hidden_dim, hidden_dim]

    return first_term - second_term


def calculate_fisher_batch(model_name, layer_idx, batch_tokens):
    """
    Calculate Fisher matrices for a batch of token sequences

    Args:
        batch_tokens: [batch_size, seq_len] - all sequences should have same length

    Returns:
        fisher_matrices: [batch_size, hidden_dim, hidden_dim]
    """
    gpt_lower = LowerLayersModel(model_name, layer_idx)
    gpt_upper = UpperLayersModel(model_name, layer_idx)

    # Process batch through lower layers
    context_vectors = gpt_lower(batch_tokens)  # [batch_size, seq_len, hidden_dim]
    gpt_lower = gpt_lower.cpu()
    torch.cuda.empty_cache()

    # Set context (all but last position) and get final latents
    gpt_upper.set_context(context_vectors[:, :-1, :])  # [batch_size, seq_len-1, hidden_dim]
    final_latents = context_vectors[:, -1, :]  # [batch_size, hidden_dim]

    # Compute gradients and probabilities
    grads, probs = gpt_upper.jacobian_batch(final_latents)

    # Compute Fisher matrices
    return fim_expected_gradient_outerproduct_batch(grads, probs)
