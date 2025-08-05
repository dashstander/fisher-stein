#!/usr/bin/env python3
"""
Script to calculate Fisher-Stein matrices for Wikipedia dataset.
Computes Fisher matrices w.r.t. penultimate layer activations and saves
both Fisher matrices and layer contributions for analysis.

Modified to include sampling from model's predictive distribution.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from fisher_stein.split_model import LowerLayersModel, UpperLayersModel
from fisher_stein.fisher_information import fim_expected_gradient_outerproduct


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_and_chunk_dataset(tokenizer, max_seq_len: int = 512, num_samples: int = 1000) -> Dict[int, List[torch.Tensor]]:
    """
    Load Wikipedia dataset, tokenize, and group by sequence length.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        num_samples: Number of samples to process
        
    Returns:
        Dictionary mapping sequence lengths to lists of token tensors
    """
    logging.info("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    # Add EOS token if it doesn't exist
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    sequences_by_length = {}
    processed = 0
    
    for example in dataset:
        if processed >= num_samples:
            break
            
        text = example['text']
        if len(text.strip()) < 50:  # Skip very short texts
            continue
            
        # Tokenize and add EOS token
        tokens = tokenizer(
            text,
            max_length=max_seq_len - 1,  # Leave room for EOS
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Add EOS token
        tokens = torch.cat([tokens, torch.tensor([tokenizer.eos_token_id])])
        seq_len = len(tokens)
        
        # Skip sequences that are too short for meaningful analysis
        if seq_len < 10:
            continue
            
        if seq_len not in sequences_by_length:
            sequences_by_length[seq_len] = []
            
        sequences_by_length[seq_len].append(tokens)
        processed += 1
        
        if processed % 100 == 0:
            logging.info(f"Processed {processed} sequences")
    
    # Log statistics
    for seq_len, sequences in sequences_by_length.items():
        logging.info(f"Length {seq_len}: {len(sequences)} sequences")
    
    return sequences_by_length


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
        context_tokens: [batch_size, seq_len] context tokens
        num_samples: Number of samples to draw per context
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        sampled_tokens: [batch_size * num_samples, seq_len + 1] tokens with sampled continuations
    """
    # Load model for sampling (we'll delete it after)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
    model.eval()
    
    batch_size, seq_len = context_tokens.shape
    all_sampled = []
    
    with torch.no_grad():
        # Process each context separately to avoid memory issues
        for i in range(batch_size):
            context = context_tokens[i:i+1].to('cuda:0')  # [1, seq_len]
            
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
            
            # Create extended sequences
            for sample_idx in sampled_indices:
                extended = torch.cat([context.cpu().squeeze(0), sample_idx.cpu().unsqueeze(0)])  # [seq_len + 1]
                all_sampled.append(extended)
    
    # Clean up model
    del model
    torch.cuda.empty_cache()
    
    return torch.stack(all_sampled)  # [batch_size * num_samples, seq_len + 1]


def calculate_fisher_for_batch_with_sampling(
    model_name: str,
    layer_idx: int,
    context_tokens: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Fisher matrices with sampling from model's predictive distribution.
    
    Args:
        model_name: Name of the model
        layer_idx: Layer index to split at (>= 2)
        context_tokens: [batch_size, seq_len] context tokens
        num_samples: Number of samples per context
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        fisher_matrices: [batch_size, hidden_dim, hidden_dim] - averaged over samples
        layer_contributions: [batch_size, num_samples, hidden_dim] - all sample contributions
        sampled_tokens: [batch_size, num_samples] - the sampled next tokens
    """
    # Sample next tokens from model's distribution
    extended_tokens = sample_next_tokens(
        model_name, context_tokens, num_samples, temperature, top_p
    )  # [batch_size * num_samples, seq_len + 1]
    
    batch_size = context_tokens.shape[0]
    
    # Extract the sampled tokens for recording
    sampled_tokens = extended_tokens[:, -1].view(batch_size, num_samples)  # [batch_size, num_samples]
    
    # Calculate Fisher matrices for all extended sequences
    gpt_lower = LowerLayersModel(model_name, layer_idx)
    gpt_upper = UpperLayersModel(model_name, layer_idx)
    
    # Process all samples through lower layers
    hidden_ult, hidden_penult = gpt_lower(extended_tokens.to('cuda:0'))
    gpt_lower = gpt_lower.cpu()
    torch.cuda.empty_cache()
    
    # Calculate layer contributions
    layer_contributions = hidden_ult[:, -1, :] - hidden_penult[:, -1, :]  # [batch_size * num_samples, hidden_dim]
    layer_contributions = layer_contributions.view(batch_size, num_samples, -1)  # [batch_size, num_samples, hidden_dim]
    
    # Set context and final latents
    context = hidden_ult[:, :-1, :]  # [batch_size * num_samples, seq_len, hidden_dim]
    final_latents = hidden_penult[:, -1, :]  # [batch_size * num_samples, hidden_dim]
    
    # Compute gradients and probabilities
    grads, probs = gpt_upper.jacobian(final_latents, context)
    
    # Compute Fisher matrices for all samples
    fisher_matrices_all = fim_expected_gradient_outerproduct(grads, probs)  # [batch_size * num_samples, hidden_dim, hidden_dim]
    
    # Reshape and average over samples
    fisher_matrices_all = fisher_matrices_all.view(batch_size, num_samples, *fisher_matrices_all.shape[1:])
    fisher_matrices = fisher_matrices_all.mean(dim=1)  # [batch_size, hidden_dim, hidden_dim]
    
    return fisher_matrices, layer_contributions, sampled_tokens


def process_sequences_by_length(
    model_name: str,
    layer_idx: int,
    sequences_by_length: Dict[int, List[torch.Tensor]],
    batch_size: int,
    num_samples: int,
    output_dir: Path,
    max_positions: int = None,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> None:
    """
    Process sequences grouped by length and save Fisher matrices.
    
    Args:
        model_name: Name of the model
        layer_idx: Layer index to split at
        sequences_by_length: Dictionary of sequences grouped by length
        batch_size: Batch size for processing (before sampling expansion)
        num_samples: Number of samples per context
        output_dir: Directory to save results
        max_positions: Maximum number of positions per sequence length to process
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    """
    
    for seq_len, sequences in sequences_by_length.items():
        logging.info(f"Processing sequences of length {seq_len} ({len(sequences)} total)")
        
        # Process in batches
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # Stack into tensor
            batch_tokens = torch.stack(batch_sequences)
            actual_batch_size = batch_tokens.shape[0]
            
            logging.info(f"  Batch {batch_idx + 1}/{num_batches}, size: {actual_batch_size}")
            logging.info(f"  Will generate {actual_batch_size * num_samples} total samples")
            
            # Process each position in the sequence (except EOS token)
            max_pos = min(seq_len - 1, max_positions) if max_positions else seq_len - 1
            
            for pos in range(max_pos):  # Don't process EOS token
                logging.info(f"    Position {pos + 1}/{max_pos}")
                
                # Get context tokens up to current position
                context_tokens = batch_tokens[:, :pos + 1]  # [batch_size, pos + 1]
                
                try:
                    fisher_matrices, layer_contributions, sampled_tokens = calculate_fisher_for_batch_with_sampling(
                        model_name, layer_idx, context_tokens, num_samples, temperature, top_p
                    )
                    
                    # Save results
                    save_data = {
                        'fisher_matrices': fisher_matrices.cpu().numpy(),
                        'layer_contributions': layer_contributions.cpu().numpy(),
                        'sampled_tokens': sampled_tokens.cpu().numpy(),
                        'metadata': {
                            'model_name': model_name,
                            'layer_idx': layer_idx,
                            'seq_len': seq_len,
                            'position': pos,
                            'batch_idx': batch_idx,
                            'batch_size': actual_batch_size,
                            'num_samples': num_samples,
                            'temperature': temperature,
                            'top_p': top_p,
                            'context_tokens': context_tokens.cpu().numpy().tolist()
                        }
                    }
                    
                    # Create filename
                    filename = f"fisher_len{seq_len}_pos{pos:03d}_batch{batch_idx:03d}_samples{num_samples}.npz"
                    filepath = output_dir / filename
                    
                    # Save as compressed numpy archive
                    np.savez_compressed(
                        filepath,
                        fisher_matrices=save_data['fisher_matrices'],
                        layer_contributions=save_data['layer_contributions'],
                        sampled_tokens=save_data['sampled_tokens'],
                        metadata=json.dumps(save_data['metadata'])
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing seq_len={seq_len}, pos={pos}, batch={batch_idx}: {e}")
                    continue
                    
                # Clear GPU memory
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Calculate Fisher-Stein matrices for Wikipedia dataset with sampling")
    parser.add_argument("--model_name", default="openai-community/gpt2", help="Model name")
    parser.add_argument("--layer_idx", type=int, default=6, help="Layer index to split at (>= 2)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (before sampling expansion)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per context")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--num_sequences", type=int, default=1000, help="Number of sequences to process")
    parser.add_argument("--max_positions", type=int, default=None, help="Max positions per sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Validate arguments
    if args.layer_idx < 2:
        raise ValueError("layer_idx must be >= 2")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'model_name': args.model_name,
        'layer_idx': args.layer_idx,
        'batch_size': args.batch_size,
        'num_samples': args.num_samples,
        'max_seq_len': args.max_seq_len,
        'num_sequences': args.num_sequences,
        'max_positions': args.max_positions,
        'temperature': args.temperature,
        'top_p': args.top_p
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration: {config}")
    logging.info(f"Effective batch size will be: {args.batch_size * args.num_samples}")
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and chunk dataset
    sequences_by_length = load_and_chunk_dataset(
        tokenizer, 
        max_seq_len=args.max_seq_len,
        num_samples=args.num_sequences
    )
    
    if not sequences_by_length:
        logging.error("No sequences loaded!")
        return
    
    # Process sequences
    logging.info("Starting Fisher matrix calculation with sampling...")
    process_sequences_by_length(
        args.model_name,
        args.layer_idx,
        sequences_by_length,
        args.batch_size,
        args.num_samples,
        output_dir,
        args.max_positions,
        args.temperature,
        args.top_p
    )
    
    logging.info("Completed Fisher matrix calculation!")
    
    # Save summary statistics
    summary = {
        'total_sequences': sum(len(seqs) for seqs in sequences_by_length.values()),
        'sequence_lengths': {str(k): len(v) for k, v in sequences_by_length.items()},
        'effective_batch_size': args.batch_size * args.num_samples,
        'config': config
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
