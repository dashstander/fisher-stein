#!/usr/bin/env python3
"""
Script to calculate Fisher-Stein matrices for Wikipedia dataset.
Computes Fisher matrices w.r.t. penultimate layer activations and saves
one file per sequence with full position-wise analysis.

Uses sliding window context with EOS attention sink and saves results
as (max_seq_len, model_dim, model_dim) tensors.
"""

import argparse
from datasets import load_dataset
import json
import logging
import numpy as np
import optree
from pathlib import Path
import torch
from transformers import AutoTokenizer, DynamicCache
from tqdm import tqdm, trange
from typing import List, Tuple, Optional

from fisher_stein.split_model import LowerLayersModel, UpperLayersModel
from fisher_stein.fisher_information import fim_expected_gradient_outerproduct


parser = argparse.ArgumentParser(description="Calculate Fisher-Stein matrices for Wikipedia dataset, one sequence at a time")
parser.add_argument("--model_name", default="openai-community/gpt2", help="Model name")
parser.add_argument("--layer_idx", type=int, default=6, help="Layer index to split at (>= 0)")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per position")
parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length to load from dataset")
parser.add_argument("--max_ctxt_length", type=int, default=128, help="Maximum context length for Jacobian calculation")
parser.add_argument("--jacrev_chunk_size", type=int, default=None, help="Chunk size for Jacobian computation")
parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to process")
parser.add_argument("--max_positions", type=int, default=None, help="Max positions per sequence")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling parameter")
parser.add_argument("--start_sequence_idx", type=int, default=0, help="Starting sequence index (for resuming)")
parser.add_argument("--device", type=str, default='cuda:0', help="Device")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_sliding_window_hidden(full_hidden: torch.Tensor, pos: int, max_ctxt_length: int) -> torch.Tensor:
    """
    Get sliding window context from hidden states.
    
    Args:
        full_hidden: [seq_len, hidden_dim] - full sequence hidden states
        pos: Current position (0-indexed)
        max_ctxt_length: Maximum context length for sliding window
        
    Returns:
        context_hidden: [context_len, hidden_dim] - sliding window context
    """
    if pos + 1 <= max_ctxt_length:
        # Use everything up to position pos (inclusive)
        return full_hidden[:pos + 1]
    else:
        # Use attention sink (position 0) + recent context
        attention_sink = full_hidden[0:1]  # [1, hidden_dim]
        recent_start = pos + 1 - (max_ctxt_length - 1)
        recent_context = full_hidden[recent_start:pos + 1]  # [max_ctxt_length-1, hidden_dim]
        return torch.cat([attention_sink, recent_context], dim=0)  # [max_ctxt_length, hidden_dim]


def get_sliding_window_cache_indices(pos: int, max_ctxt_length: int) -> Tuple[List[int], int]:
    """
    Get the indices for sliding window cache slicing.
    
    Args:
        pos: Current position (0-indexed)
        max_ctxt_length: Maximum context length
        
    Returns:
        indices: List of position indices to keep in cache
        context_length: Actual context length after windowing
    """
    if pos + 1 <= max_ctxt_length:
        # Use all positions up to pos
        indices = list(range(pos + 1))
        context_length = pos + 1
    else:
        # Use attention sink (0) + recent context
        recent_start = pos + 1 - (max_ctxt_length - 1)
        indices = [0] + list(range(recent_start, pos + 1))
        context_length = max_ctxt_length
    
    return indices, context_length


def slice_cache_tensor_for_sliding_window(cache_tensor: torch.Tensor, pos: int, max_ctxt_length: int) -> torch.Tensor:
    """
    Slice a single cache tensor for sliding window.
    
    Args:
        cache_tensor: [batch, heads, seq_len, head_dim] - cache tensor from one layer
        pos: Current position
        max_ctxt_length: Maximum context length
        
    Returns:
        sliced_tensor: [batch, heads, context_len, head_dim] - sliced cache tensor
    """    
    if pos + 1 <= max_ctxt_length:
        # Simple case: just truncate
        return cache_tensor[:, :, :pos + 1, :]
    else:
        # Complex case: attention sink + recent context
        attention_sink = cache_tensor[:, :, 0:1, :]  # [batch, heads, 1, head_dim]
        recent_start = pos + 1 - (max_ctxt_length - 1)
        recent_context = cache_tensor[:, :, recent_start:pos + 1, :]  # [batch, heads, max_ctxt_length-1, head_dim]
        return torch.cat([attention_sink, recent_context], dim=2)


def get_sliding_window_cache(full_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
                           pos: int, max_ctxt_length: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Get sliding window cache using optree.tree_map.
    
    Args:
        full_kv_cache: Legacy cache format [(keys, values), ...] for each layer
        pos: Current position
        max_ctxt_length: Maximum context length
        
    Returns:
        windowed_cache: Legacy cache format with sliding window applied
    """
    def slice_fn(tensor):
        return slice_cache_tensor_for_sliding_window(tensor, pos, max_ctxt_length)
    
    return optree.tree_map(slice_fn, full_kv_cache)


def expand_cache_for_batch(cache_legacy: List[Tuple[torch.Tensor, torch.Tensor]], 
                          batch_size: int) -> DynamicCache:
    """
    Expand a legacy cache to handle a batch of sequences.
    
    Args:
        cache_legacy: Legacy cache format [(keys, values), ...] 
        batch_size: Number of sequences in batch
        
    Returns:
        batch_cache: DynamicCache ready for batch processing
    """
    def expand_tensor(tensor):
        # tensor is [1, heads, seq_len, head_dim], expand to [batch_size, heads, seq_len, head_dim]
        return tensor.expand(batch_size, -1, -1, -1).contiguous()
    
    expanded_cache = optree.tree_map(expand_tensor, cache_legacy)
    return DynamicCache.from_legacy_cache(expanded_cache)


def sample_next_tokens_at_position(
    model,
    context_hidden: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> torch.Tensor:
    """
    Sample next tokens from the model at a specific position.
    
    Args:
        model: Upper layers model with calculate_logits method
        context_hidden: [context_len, hidden_dim] - context hidden states
        num_samples: Number of samples to draw
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        sampled_token_ids: [num_samples] - sampled next tokens
    """
    with torch.no_grad():
        # Get logits for next token using the last position of context
        # context_hidden is [context_len, hidden_dim], we want the last position
        final_hidden = context_hidden[-1:].unsqueeze(0).to('cuda:0')  # [1, 1, hidden_dim]
        
        # Get logits from upper model
        logits = model.calculate_logits(final_hidden)  # Should return [1, vocab_size]
        next_token_logits = logits[0] / temperature  # [vocab_size]
        
        # Apply nucleus sampling if specified
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample multiple tokens
        probs = torch.softmax(next_token_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples, replacement=True)  # [num_samples]
    
    return sampled_indices.cpu()


def load_sequences(tokenizer, max_seq_len: int = 512, num_sequences: int = 1000) -> List[Tuple[torch.Tensor, str, str]]:
    """
    Load Wikipedia sequences with EOS attention sink.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length (including EOS)
        num_sequences: Number of sequences to load
        
    Returns:
        List of (tokens_tensor, text_preview, title) tuples
    """
    logging.info(f"Loading {num_sequences} Wikipedia sequences...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    # Ensure EOS token exists
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    # For GPT-2, use EOS as attention sink at beginning
    attention_sink_id = tokenizer.eos_token_id
    
    sequences = []
    processed = 0
    
    for example in dataset:
        if len(sequences) >= num_sequences:
            break
            
        text = example['text']
        title = example.get('title', 'Unknown')
        
        if len(text.strip()) < 100:  # Skip very short texts
            continue
            
        # Tokenize with room for attention sink at beginning
        tokens = tokenizer(
            text,
            max_length=max_seq_len - 1,  # Leave room for attention sink
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Prepend attention sink (EOS token)
        tokens_with_sink = torch.cat([torch.tensor([attention_sink_id]), tokens])
        
        # Skip sequences that are too short for meaningful analysis
        if len(tokens_with_sink) < 20:
            continue
        
        # Create text preview for metadata
        text_preview = text[:200].replace('\n', ' ').strip()
        
        sequences.append((tokens_with_sink, text_preview, title))
        processed += 1
        
        if processed % 100 == 0:
            logging.info(f"Loaded {processed} sequences")
    
    logging.info(f"Successfully loaded {len(sequences)} sequences")
    return sequences

def calculate_fisher_at_position(
    pos: int,
    gpt_lower: LowerLayersModel,
    gpt_upper: UpperLayersModel,
    full_hidden: torch.Tensor,
    full_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    max_ctxt_length: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float,
    top_p: float,
    device: str
) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    """
    Calculate Fisher matrix for a single position.
    
    Returns:
        fisher_matrix: [hidden_dim, hidden_dim]
        metadata: dict with position info
        sampled_tokens: [num_samples]
    """
    # Get sliding window context for this position
    context_hidden = get_sliding_window_hidden(full_hidden, pos, max_ctxt_length)
    
    # Sample next tokens based on this position's context
    sampled_token_ids = sample_next_tokens_at_position(
        gpt_upper, context_hidden, num_samples, temperature, top_p
    )
    
    # Get sliding window cache for this position
    context_cache_legacy = get_sliding_window_cache(full_kv_cache, pos, max_ctxt_length)
    
    # Move to device and expand for batch processing
    context_cache_on_device = optree.tree_map(lambda x: x.to(device), context_cache_legacy)
    batch_context_cache = expand_cache_for_batch(context_cache_on_device, num_samples)
    
    # Process each sampled token with the context cache
    final_latents_batch = gpt_lower.forward(
        sampled_token_ids.unsqueeze(-1).to(device),  # [num_samples, 1]
        past_key_value=batch_context_cache,
        use_cache=True
    )[:, -1, :]  # [num_samples, hidden_dim] - just the new tokens
    
    # Use the sliding window context for upper layers too
    context_for_upper = context_hidden.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
    
    # Compute gradients and probabilities
    grads, probs = gpt_upper.jacobian(final_latents_batch, context_for_upper, jacrev_chunk_size)
    
    # Compute Fisher matrices for all samples
    fisher_matrices_samples = fim_expected_gradient_outerproduct(grads, probs)
    
    # Average over samples
    fisher_avg = fisher_matrices_samples.mean(dim=0).cpu()  # [hidden_dim, hidden_dim]
    
    # Store metadata
    effective_context_len = len(context_hidden)
    metadata = {
        'position': pos,
        'effective_context_len': effective_context_len,
        'uses_sliding_window': pos + 1 > max_ctxt_length,
        'num_samples_used': len(sampled_token_ids),
        'sample_token_ids': sampled_token_ids.tolist()
    }
    
    # All GPU tensors will be automatically cleaned up when this function returns
    return fisher_avg, metadata, sampled_token_ids


def calculate_fisher_for_single_sequence(
    lower_model,
    upper_model,
    sequence_tokens: torch.Tensor,
    max_ctxt_length: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_positions: int = None,
    device: str = 'cuda:0'
) -> Tuple[torch.Tensor, List[dict], List[torch.Tensor]]:
    
    seq_len = len(sequence_tokens)
    max_pos = min(seq_len - 1, max_positions) if max_positions else seq_len - 1
    
    # 1. Process full sequence once to get hidden states and cache
    full_hidden, full_kv_cache = lower_model.forward_with_cache(sequence_tokens.unsqueeze(0).to(device))
    full_hidden = full_hidden.squeeze(0).cpu()  # [seq_len, hidden_dim]
    full_kv_cache = optree.tree_map(lambda x: x.cpu(), full_kv_cache)
    hidden_dim = full_hidden.shape[-1]
    
    # 2. Initialize results
    fisher_matrices = torch.zeros(max_pos, hidden_dim, hidden_dim, dtype=torch.float32)
    position_metadata = []
    sampled_tokens_list = []
    
    # 3. Process each position
    for pos in trange(max_pos):
        fisher_matrix, metadata, sampled_tokens = calculate_fisher_at_position(
            pos, lower_model, upper_model, full_hidden, full_kv_cache,
            max_ctxt_length, num_samples, jacrev_chunk_size,
            temperature, top_p, device
        )
        
        fisher_matrices[pos] = fisher_matrix
        position_metadata.append(metadata)
        sampled_tokens_list.append(sampled_tokens)
            
    return fisher_matrices, position_metadata, sampled_tokens_list


def process_sequences(
    model_name: str,
    layer_idx: int,
    sequences: List[Tuple[torch.Tensor, str, str]],
    output_dir: Path,
    max_ctxt_length: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_positions: Optional[int] = None,
    start_sequence_idx: int = 0,
    device: str = 'cuda:0'
) -> None:
    """
    Process all sequences and save results.
    """
    for seq_idx, (sequence_tokens, text_preview, title) in tqdm(enumerate(sequences[start_sequence_idx:]), desc="Processing sequences"):
        actual_seq_idx = start_sequence_idx + seq_idx
        logging.info(f"Processing sequence {actual_seq_idx + 1}/{len(sequences)}: {title}")
        
        try:
            # Calculate Fisher matrices for this sequence
            fisher_matrices, position_metadata, sampled_tokens = calculate_fisher_for_single_sequence(
                model_name, layer_idx, sequence_tokens, max_ctxt_length, 
                num_samples, jacrev_chunk_size, temperature, top_p, max_positions, device
            )
            
            # Create sequence metadata
            sequence_metadata = {
                'sequence_id': actual_seq_idx,
                'model_name': model_name,
                'layer_idx': layer_idx,
                'title': title,
                'text_preview': text_preview,
                'sequence_length': len(sequence_tokens),
                'max_ctxt_length': max_ctxt_length,
                'num_samples': num_samples,
                'temperature': temperature,
                'top_p': top_p,
                'positions_analyzed': len(position_metadata),
                'full_sequence_tokens': sequence_tokens.tolist(),
                'position_metadata': position_metadata
            }
            
            # Save results
            filename = f"fisher_seq_{actual_seq_idx:06d}.npz"
            filepath = output_dir / filename
            
            np.savez_compressed(
                filepath,
                fisher_matrices=fisher_matrices.numpy(),
                sampled_tokens=[tokens.numpy() for tokens in sampled_tokens],
                metadata=json.dumps(sequence_metadata, indent=2)
            )
            
            logging.info(f"Saved sequence {actual_seq_idx + 1} to {filename}")
            
        except Exception as e:
            logging.error(f"Failed to process sequence {actual_seq_idx + 1}: {e}")
            continue


def main():
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate arguments
    if args.layer_idx < 0:
        raise ValueError("layer_idx must be >= 0")
    
    if args.max_ctxt_length > args.max_seq_len:
        raise ValueError("max_ctxt_length cannot be greater than max_seq_len")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'model_name': args.model_name,
        'layer_idx': args.layer_idx,
        'num_samples': args.num_samples,
        'max_seq_len': args.max_seq_len,
        'max_ctxt_length': args.max_ctxt_length,
        'jacrev_chunk_size': args.jacrev_chunk_size,
        'num_sequences': args.num_sequences,
        'max_positions': args.max_positions,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'start_sequence_idx': args.start_sequence_idx,
        'device': args.device,

    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration: {config}")
    logging.info(f"Processing sequences one at a time with {args.num_samples} samples per position")
    logging.info(f"Using sliding window context with max length: {args.max_ctxt_length}")
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load sequences
    sequences = load_sequences(
        tokenizer, 
        max_seq_len=args.max_seq_len,
        num_sequences=args.num_sequences
    )
    
    if not sequences:
        logging.error("No sequences loaded!")
        return
    
    # Process sequences
    logging.info("Starting Fisher matrix calculation...")
    process_sequences(
        args.model_name,
        args.layer_idx,
        sequences,
        output_dir,
        args.max_ctxt_length,
        args.num_samples,
        args.jacrev_chunk_size,
        args.temperature,
        args.top_p,
        args.max_positions,
        args.start_sequence_idx,
        args.device
    )
    
    logging.info("Completed Fisher matrix calculation!")
    
    # Save summary statistics
    summary = {
        'total_sequences_processed': len(sequences),
        'sequences_analyzed': f"{args.start_sequence_idx} to {min(args.start_sequence_idx + len(sequences), args.num_sequences)}",
        'config': config
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
