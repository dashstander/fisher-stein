import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.nn.functional import softmax, log_softmax


class LowerLayersModel(nn.Module):
    """Model that only uses the lower layers up to layer_idx"""
    def __init__(self, model_name, layer_idx, device="cuda"):
        super().__init__()

        # Load original model temporarily
        original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Extract only the layers we need
        self.embedding = original_model.transformer.wte
        self.position_embedding = original_model.transformer.wpe

        # Only keep layers up to layer_idx
        self.layers = nn.ModuleList([
            original_model.transformer.h[i]
            for i in range(layer_idx)
        ])

        # Delete the original model to free memory
        del original_model
        torch.cuda.empty_cache()

        # Move to device
        self.to(device)
        self.device = device

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        """Compute representations up to layer_idx
        Args:
            input_ids: [batch_size, seq_len] or [seq_len]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
        """
        with torch.no_grad():
            # Handle both batched and single inputs
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False

            batch_size, seq_len = input_ids.shape

            # Get embeddings
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            # Compute embeddings
            token_embeds = self.embedding(input_ids)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds

            # Process through layers
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]

            if squeeze_output:
                hidden_states = hidden_states.squeeze(0)

            return hidden_states


class UpperLayersModel(nn.Module):
    """Model that only uses the upper layers from layer_idx onwards"""
    def __init__(self, model_name, layer_idx, device="cuda"):
        super().__init__()

        # Load original model temporarily
        original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Only keep layers from layer_idx onwards
        self.layers = nn.ModuleList([
            original_model.transformer.h[i]
            for i in range(layer_idx, len(original_model.transformer.h))
        ])

        # Output components
        self.ln_f = original_model.transformer.ln_f
        self.lm_head = original_model.lm_head

        # Delete the original model to free memory
        del original_model
        torch.cuda.empty_cache()

        # Move to device
        self.to(device)
        self.device = device

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, final_latent):
        """
        Forward pass with stored context and final latent(s)
        Args:
            final_latent: [batch_size, hidden_size] or [hidden_size]
        Returns:
            log_probs: [batch_size, vocab_size] or [vocab_size]
        """
        # Ensure context has been set
        if self.context_vectors is None:
            raise ValueError("Context vectors have not been set. Call set_context first.")

        # Handle both batched and single inputs
        if final_latent.dim() == 1:
            final_latent = final_latent.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = final_latent.shape[0]

        # Ensure context is batched correctly
        if self.context_vectors.dim() == 2:
            # Context is [context_len, hidden_dim], expand to batch
            context = self.context_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Context is already [batch_size, context_len, hidden_dim]
            context = self.context_vectors

        # Combine context and target latent
        latents = torch.cat([context, final_latent.unsqueeze(1)], dim=1)

        # Process through remaining layers
        hidden_states = latents
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]

        # Final layer norm and LM head
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Get final position logits
        final_logits = logits[:, -1]  # [batch_size, vocab_size]

        log_probs = log_softmax(final_logits, dim=-1)

        if squeeze_output:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def jacobian(self, final_latents, context, cache=None):
        """Get gradients for all tokens efficiently via chain rule - batched version
        Args:
            final_latents: [batch_size, hidden_dim]
        Returns:
            J_log_probs: [batch_size, vocab_size, hidden_dim]
            probs: [batch_size, vocab_size]
        """

        def hidden_states_fn_batch(x, context):
            """
            Args:
                x: [batch_size, hidden_dim]
            Returns:
                [batch_size, hidden_dim]
            """
            latents = torch.cat([context, x.unsqueeze(0)], dim=0).unsqueeze(0)
            hidden_states = latents
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
            hidden_states = self.ln_f(hidden_states)
            return hidden_states.squeeze()[-1]

        # Compute jacobian: [batch_size, hidden_dim, hidden_dim]
        J_hidden = torch.vmap(torch.func.jacrev(hidden_states_fn_batch, argnums=0), in_dims=0)(final_latents, context)

        # Chain through unembedding: [batch_size, vocab_size, hidden_dim]
        # J_logits[b, v, h] = sum_k W_unembed[v, k] * J_hidden[b, k, h]
        J_logits = torch.einsum('vk,bkh->bvh', self.lm_head.weight, J_hidden)

        # Get current logits and probabilities
        with torch.no_grad():
            hidden_final = torch.vmap(hidden_states_fn_batch)(final_latents, context)  # [batch_size, hidden_dim]
            logits = self.lm_head(hidden_final)  # [batch_size, vocab_size]
            probs = torch.softmax(logits, dim=-1)  # [batch_size, vocab_size]

        def log_softmax_fn(x):
            return log_softmax(x, dim=-1)

        # Apply JVP for log_softmax transformation
        # For each batch and each hidden dimension
        def apply_jvp_single(args):
            logits_single, J_logits_single = args
            return torch.vmap(
                lambda col: torch.func.jvp(log_softmax_fn, (logits_single,), (col,))[1],
                in_dims=1
            )(J_logits_single).T

        # Vmap over batch dimension
        J_log_probs = torch.vmap(apply_jvp_single)((logits, J_logits))

        return J_log_probs, probs


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
    context_vectors = gpt_lower(batch_tokens.to('cuda:0'))  # [batch_size, seq_len, hidden_dim]
    gpt_lower = gpt_lower.cpu()
    torch.cuda.empty_cache()

    # Set context (all but last position) and get final latents
    gpt_upper.set_context(context_vectors[:, :-1, :])  # [batch_size, seq_len-1, hidden_dim]
    final_latents = context_vectors[:, -1, :]  # [batch_size, hidden_dim]

    # Compute gradients and probabilities
    grads, probs = gpt_upper.jacobian_batch(final_latents)

    # Compute Fisher matrices
    return fim_expected_gradient_outerproduct_batch(grads, probs)




