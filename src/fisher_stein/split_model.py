import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.nn.functional import log_softmax


class LowerLayersModel(nn.Module):
    """Model that only uses the lower layers up to layer_idx"""
    def __init__(self, model_name, layer_idx, device="cuda"):
        super().__init__()

        assert layer_idx > 0

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
        """Compute representations up to `layer_idx`. Returns hidden states from `layer_idx` and `layer_idx - 1` so that they can be used for analysis.
            input_ids: [batch_size, seq_len] or [seq_len]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
            prev_states: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
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
            prev_states = torch.empty_like(hidden_states)

            # Process through layers
            for layer in self.layers:
                prev_states = hidden_states
                hidden_states = layer(hidden_states)[0]

            if squeeze_output:
                prev_states = prev_states.squeeze(0)
                hidden_states = hidden_states.squeeze(0)

            return hidden_states, prev_states


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

    def forward(self, final_latent, context, kv_cache=None, use_cache=False):
        """
        Forward pass with stored context and final latent(s)
        Args:
            final_latent: [batch_size, hidden_size] or [hidden_size]
        Returns:
            log_probs: [batch_size, vocab_size] or [vocab_size]
        """
        hidden_states = torch.cat([context, final_latent.unsqueeze(0)], dim=0).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.ln_f(hidden_states)
        return hidden_states.squeeze()[-1]


    def jacobian(self, final_latents, context):
        """Get Jacobian of the model output, the gradient of **each** logit, calculated with $O(d_{\text{vocab}})$ backward passes over the "upper" layers.
        Args:
            final_latents: [batch_size, hidden_dim]
            context: [batch_size, seq_len, hidden_dim]
        Returns:
            J_log_probs: [batch_size, vocab_size, hidden_dim]
            probs: [batch_size, vocab_size]
        """

        # Compute jacobian: [batch_size, hidden_dim, hidden_dim]
        J_hidden = torch.vmap(torch.func.jacrev(self.forward, argnums=0), in_dims=0)(final_latents, context)

        # Chain through unembedding: [batch_size, vocab_size, hidden_dim]
        # J_logits[b, v, h] = sum_k W_unembed[v, k] * J_hidden[b, k, h]
        J_logits = torch.einsum('vk,bkh->bvh', self.lm_head.weight, J_hidden)

        # Get current logits and probabilities
        with torch.no_grad():
            hidden_final = torch.vmap(self.forward)(final_latents, context)  # [batch_size, hidden_dim]
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
