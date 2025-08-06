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
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from fisher_stein.split_model import LowerLayersModel, UpperLayersModel
from fisher_stein.fisher_information import fim_expected_gradient_outerproduct


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def sample_next_tokens(
    model_name: str,
    context_tokens: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> torch.Tensor:
    """
    Sample next tokens from the model's predictive distribution.
    
    Args:
        model_name: Name of the model
        context_tokens: [seq_len] context tokens for single sequence
        num_samples: Number of samples to draw
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        sampled_token_ids: [num_samples] sampled next tokens
    """
    # Load model for sampling (we'll delete it after)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
    model.eval()
    
    with torch.no_grad():
        context = context_tokens.unsqueeze(0).to('cuda:0')  # [1, seq_len]
        
        # Get logits for next token
        outputs = model(context)
        next_token_logits = outputs.logits[0, -1, :] / temperature  # [vocab_size]
        
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
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    return sampled_indices.cpu()


def get_sliding_window_context(full_hidden_states: torch.Tensor, position: int, max_ctxt_length: int) -> torch.Tensor:
    """
    Extract sliding window context that always includes attention sink (position 0) 
    plus recent context up to max_ctxt_length.
    
    Args:
        full_hidden_states: [seq_len, hidden_dim] - full sequence hidden states
        position: Current position we're predicting from
        max_ctxt_length: Maximum context length (including attention sink)
        
    Returns:
        context_window: [context_len, hidden_dim] - context for this position
    """
    if position + 1 <= max_ctxt_length:
        # Early positions: use all context up to current position
        return full_hidden_states[:position + 1]  # [0, 1, ..., position]
    else:
        # Later positions: sliding window with attention sink
        # Always include position 0 (attention sink) + recent context
        attention_sink = full_hidden_states[0:1]  # [1, hidden_dim]
        recent_context_start = position + 1 - (max_ctxt_length - 1)
        recent_context = full_hidden_states[recent_context_start:position + 1]  # [context_len-1, hidden_dim]
        return torch.cat([attention_sink, recent_context], dim=0)  # [max_ctxt_length, hidden_dim]


def get_sliding_window_tokens(full_tokens: torch.Tensor, position: int, max_ctxt_length: int) -> torch.Tensor:
    """
    Get token-level sliding window context matching the hidden state window.
    """
    if position + 1 <= max_ctxt_length:
        return full_tokens[:position + 1]
    else:
        attention_sink = full_tokens[0:1]
        recent_context_start = position + 1 - (max_ctxt_length - 1)
        recent_context = full_tokens[recent_context_start:position + 1]
        return torch.cat([attention_sink, recent_context], dim=0)


def calculate_fisher_for_single_sequence(
    model_name: str,
    layer_idx: int,
    sequence_tokens: torch.Tensor,
    max_ctxt_length: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_positions: Optional[int] = None
) -> Tuple[torch.Tensor, List[Dict], List[torch.Tensor]]:
    """
    Calculate Fisher matrices for all positions in a single sequence.
    
    Returns:
        fisher_matrices: [seq_len, hidden_dim, hidden_dim] - Fisher matrix for each position
        position_metadata: List of metadata dicts for each position
        sampled_tokens: List of [num_samples] tensors for each position
    """
    seq_len = len(sequence_tokens)
    max_pos = min(seq_len - 1, max_positions) if max_positions else seq_len - 1
    
    logging.info(f"Processing sequence of length {seq_len}, analyzing {max_pos} positions")
    
    # Process full sequence through lower layers once
    gpt_lower = LowerLayersModel(model_name, layer_idx)
    full_hidden_ult, full_hidden_penult = gpt_lower(sequence_tokens.unsqueeze(0).to('cuda:0'))
    full_hidden_ult = full_hidden_ult.squeeze(0).cpu()  # [seq_len, hidden_dim]
    full_hidden_penult = full_hidden_penult.squeeze(0).cpu()  # [seq_len, hidden_dim]
    hidden_dim = full_hidden_ult.shape[1]
    
    del gpt_lower
    torch.cuda.empty_cache()
    
    # Initialize results
    fisher_matrices = torch.zeros(max_pos, hidden_dim, hidden_dim, dtype=torch.float32)
    position_metadata = []
    sampled_tokens_list = []
    
    # Load upper model
    gpt_upper = UpperLayersModel(model_name, layer_idx)
    
    # Process each position
    for pos in tqdm(range(max_pos), desc="Computing Fisher matrices"):
        try:
            # Get sliding window context for this position
            context_hidden = get_sliding_window_context(full_hidden_ult, pos, max_ctxt_length)
            final_latent = full_hidden_penult[pos]
            context_tokens = get_sliding_window_tokens(sequence_tokens, pos, max_ctxt_length)
            
            # Sample next tokens for this context
            sampled_token_ids = sample_next_tokens(
                model_name, context_tokens, num_samples, temperature, top_p
            )
            sampled_tokens_list.append(sampled_token_ids)
            
            # Create extended sequences with sampled tokens
            extended_sequences = []
            for sample_id in sampled_token_ids:
                extended_seq = torch.cat([context_tokens, sample_id.unsqueeze(0)])
                extended_sequences.append(extended_seq)
            
            # Process extended sequences through lower layers to get final states
            max_extended_len = max(len(seq) for seq in extended_sequences)
            padded_sequences = []
            
            for seq in extended_sequences:
                if len(seq) < max_extended_len:
                    # Pad with attention sink tokens at the beginning
                    pad_size = max_extended_len - len(seq)
                    attention_sink_id = seq[0].item()
                    padding = torch.full((pad_size,), attention_sink_id, dtype=seq.dtype)
                    padded_seq = torch.cat([seq[:1], padding, seq[1:]], dim=0)
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            
            batch_extended = torch.stack(padded_sequences).to('cuda:0')  # [num_samples, max_len]
            
            # Get hidden states for extended sequences
            extended_hidden_ult, extended_hidden_penult = gpt_lower.__class__(model_name, layer_idx)(batch_extended)
            
            # Extract contexts and final latents
            contexts_batch = extended_hidden_ult[:, :-1, :]  # [num_samples, context_len, hidden_dim]
            final_latents_batch = extended_hidden_penult[:, -1, :]  # [num_samples, hidden_dim]
            
            # Compute gradients and probabilities
            grads, probs = gpt_upper.jacobian(final_latents_batch, contexts_batch, jacrev_chunk_size)
            
            # Compute Fisher matrices for all samples
            fisher_matrices_samples = fim_expected_gradient_outerproduct(grads, probs)  # [num_samples, hidden_dim, hidden_dim]
            
            # Average over samples
            fisher_avg = fisher_matrices_samples.mean(dim=0)  # [hidden_dim, hidden_dim]
            fisher_matrices[pos] = fisher_avg.cpu()
            
            # Store metadata for this position
            effective_context_len = len(context_tokens)
            position_metadata.append({
                'position': pos,
                'effective_context_len': effective_context_len,
                'uses_sliding_window': pos + 1 > max_ctxt_length,
                'context_tokens': context_tokens.tolist(),
                'num_samples_used': len(sampled_token_ids),
                'sample_token_ids': sampled_token_ids.tolist()
            })
            
        except Exception as e:
            logging.error(f"Error processing position {pos}: {e}")
            # Fill with zeros for this position
            position_metadata.append({
                'position': pos,
                'error': str(e),
                'effective_context_len': 0,
                'uses_sliding_window': False
            })
            sampled_tokens_list.append(torch.tensor([]))
            continue
        
        # Clean up GPU memory periodically
        if pos % 10 == 0:
            torch.cuda.empty_cache()
    
    # Clean up
    del gpt_upper
    torch.cuda.empty_cache()
    
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
    top_p: float = 0.9,
    max_positions: Optional[int] = None,
    start_sequence_idx: int = 0
) -> None:
    """
    Process all sequences and save results.
    """
    for seq_idx, (sequence_tokens, text_preview, title) in enumerate(tqdm(sequences[start_sequence_idx:], 
                                                                          desc="Processing sequences")):
        actual_seq_idx = start_sequence_idx + seq_idx
        logging.info(f"Processing sequence {actual_seq_idx + 1}/{len(sequences)}: {title}")
        
        try:
            # Calculate Fisher matrices for this sequence
            fisher_matrices, position_metadata, sampled_tokens = calculate_fisher_for_single_sequence(
                model_name, layer_idx, sequence_tokens, max_ctxt_length, 
                num_samples, jacrev_chunk_size, temperature, top_p, max_positions
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
    parser = argparse.ArgumentParser(description="Calculate Fisher-Stein matrices for Wikipedia dataset, one sequence at a time")
    parser.add_argument("--model_name", default="openai-community/gpt2", help="Model name")
    parser.add_argument("--layer_idx", type=int, default=6, help="Layer index to split at (>= 2)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per position")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length to load from dataset")
    parser.add_argument("--max_ctxt_length", type=int, default=128, help="Maximum context length for Jacobian calculation")
    parser.add_argument("--jacrev_chunk_size", type=int, default=128, help="Chunk size for Jacobian computation")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to process")
    parser.add_argument("--max_positions", type=int, default=None, help="Max positions per sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--start_sequence_idx", type=int, default=0, help="Starting sequence index (for resuming)")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate arguments
    if args.layer_idx < 2:
        raise ValueError("layer_idx must be >= 2")
    
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
        'start_sequence_idx': args.start_sequence_idx
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
        args.start_sequence_idx
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
