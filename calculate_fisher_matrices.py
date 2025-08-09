#!/usr/bin/env python3
"""
Script to calculate Fisher-Stein matrices for Wikipedia dataset.
Computes Fisher matrices w.r.t. penultimate layer activations and saves
one file per sequence with full position-wise analysis.

Uses sliding window context with EOS attention sink and saves results
as (max_seq_len, model_dim, model_dim) tensors.
"""

import argparse
import boto3
from botocore.exceptions import ClientError
from datasets import load_dataset
import io
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


parser = argparse.ArgumentParser(
    description="Calculate Fisher-Stein matrices for Wikipedia dataset, one sequence at a time"
)
parser.add_argument("--model_name", default="openai-community/gpt2", help="Model name")
parser.add_argument(
    "--layer_idx", type=int, default=6, help="Layer index to split at (>= 0)"
)
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument(
    "--num_samples", type=int, default=50, help="Number of samples per position"
)
parser.add_argument(
    "--max_seq_len",
    type=int,
    default=512,
    help="Maximum sequence length to load from dataset",
)
parser.add_argument(
    "--max_ctxt_len",
    type=int,
    default=128,
    help="Maximum context length for Jacobian calculation",
)
parser.add_argument(
    "--jacrev_chunk_size",
    type=int,
    default=None,
    help="Chunk size for Jacobian computation",
)
parser.add_argument(
    "--num_sequences", type=int, default=100, help="Number of sequences to process"
)
parser.add_argument(
    "--max_positions", type=int, default=None, help="Max positions per sequence"
)
parser.add_argument(
    "--position_skip",
    type=int,
    default=1,
    help="Skip positions - calculate Fisher matrices every k positions starting at position k (default: 1, process all positions)",
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Sampling temperature"
)
parser.add_argument(
    "--top_p", type=float, default=1.0, help="Nucleus sampling parameter"
)
parser.add_argument(
    "--start_sequence_idx",
    type=int,
    default=0,
    help="Starting sequence index (for resuming)",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument(
    "--s3_bucket",
    type=str,
    default=None,
    help="S3 bucket name for storing results (optional, saves locally if not provided)",
)
parser.add_argument(
    "--s3_key_prefix",
    type=str,
    default="fisher-stein-results/",
    help="S3 key prefix for organizing results (default: 'fisher-stein-results/')",
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def upload_to_s3(
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
) -> bool:
    """
    Upload data to S3.

    Args:
        data: Bytes to upload
        bucket: S3 bucket name
        key: S3 object key
        content_type: Content type for the object

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = boto3.client("s3")
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=data, ContentType=content_type
        )
        logging.info(f"Successfully uploaded to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error uploading to S3: {e}")
        return False


def save_results(
    fisher_matrices: torch.Tensor,
    sampled_tokens: List[torch.Tensor],
    sequence_metadata: dict,
    filename: str,
    output_dir: Path,
    s3_bucket: Optional[str] = None,
    s3_key_prefix: str = "fisher-stein-results/",
) -> None:
    """
    Save results directly to S3 if bucket specified, otherwise save locally.

    Args:
        fisher_matrices: Fisher matrices tensor
        sampled_tokens: List of sampled token tensors
        sequence_metadata: Metadata dictionary
        filename: Base filename (e.g., "fisher_seq_000001.npz")
        output_dir: Local output directory (used only if no S3 bucket)
        s3_bucket: Optional S3 bucket name for direct upload
        s3_key_prefix: S3 key prefix for organization
    """
    if s3_bucket:
        # Save directly to S3 using in-memory buffer
        s3_key = s3_key_prefix + filename

        # Create NPZ data in memory
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            fisher_matrices=fisher_matrices.numpy(),
            sampled_tokens=[tokens.numpy() for tokens in sampled_tokens],
            metadata=json.dumps(sequence_metadata, indent=2),
        )
        buffer.seek(0)

        if upload_to_s3(
            buffer.getvalue(), s3_bucket, s3_key, "application/octet-stream"
        ):
            logging.info(f"Saved {filename} directly to S3")
        else:
            logging.error(f"Failed to save {filename} to S3")
            raise RuntimeError(f"S3 upload failed for {filename}")
    else:
        # Save locally as fallback
        local_filepath = output_dir / filename
        np.savez_compressed(
            local_filepath,
            fisher_matrices=fisher_matrices.numpy(),
            sampled_tokens=[tokens.numpy() for tokens in sampled_tokens],
            metadata=json.dumps(sequence_metadata, indent=2),
        )
        logging.info(f"Saved locally to {filename}")


def save_json_results(
    data: dict,
    filename: str,
    output_dir: Path,
    s3_bucket: Optional[str] = None,
    s3_key_prefix: str = "fisher-stein-results/",
) -> None:
    """
    Save JSON data directly to S3 if bucket specified, otherwise save locally.
    """
    if s3_bucket:
        # Save directly to S3
        s3_key = s3_key_prefix + filename
        json_data = json.dumps(data, indent=2).encode("utf-8")

        if upload_to_s3(json_data, s3_bucket, s3_key, "application/json"):
            logging.info(f"Saved {filename} directly to S3")
        else:
            logging.error(f"Failed to save {filename} to S3")
            raise RuntimeError(f"S3 upload failed for {filename}")
    else:
        # Save locally as fallback
        local_filepath = output_dir / filename
        with open(local_filepath, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved {filename} locally")


def get_sliding_window_hidden(
    full_hidden: torch.Tensor, pos: int, max_ctxt_len: int
) -> torch.Tensor:
    """
    Get sliding window context from hidden states.

    Args:
        full_hidden: [seq_len, hidden_dim] - full sequence hidden states
        pos: Current position (0-indexed)
        max_ctxt_len: Maximum context length for sliding window

    Returns:
        context_hidden: [context_len, hidden_dim] - sliding window context
    """
    if pos + 1 <= max_ctxt_len:
        # Use everything up to position pos (inclusive)
        return full_hidden[: pos + 1]
    else:
        # Use attention sink (position 0) + recent context
        attention_sink = full_hidden[0:1]  # [1, hidden_dim]
        recent_start = pos + 1 - (max_ctxt_len - 1)
        recent_context = full_hidden[
            recent_start : pos + 1
        ]  # [max_ctxt_len-1, hidden_dim]
        return torch.cat(
            [attention_sink, recent_context], dim=0
        )  # [max_ctxt_len, hidden_dim]


def get_sliding_window_cache_indices(
    pos: int, max_ctxt_len: int
) -> Tuple[List[int], int]:
    """
    Get the indices for sliding window cache slicing.

    Args:
        pos: Current position (0-indexed)
        max_ctxt_len: Maximum context length

    Returns:
        indices: List of position indices to keep in cache
        context_length: Actual context length after windowing
    """
    if pos + 1 <= max_ctxt_len:
        # Use all positions up to pos
        indices = list(range(pos + 1))
        context_length = pos + 1
    else:
        # Use attention sink (0) + recent context
        recent_start = pos + 1 - (max_ctxt_len - 1)
        indices = [0] + list(range(recent_start, pos + 1))
        context_length = max_ctxt_len

    return indices, context_length


def slice_cache_tensor_for_sliding_window(
    cache_tensor: torch.Tensor, pos: int, max_ctxt_len: int
) -> torch.Tensor:
    """
    Slice a single cache tensor for sliding window.

    Args:
        cache_tensor: [batch, heads, seq_len, head_dim] - cache tensor from one layer
        pos: Current position
        max_ctxt_len: Maximum context length

    Returns:
        sliced_tensor: [batch, heads, context_len, head_dim] - sliced cache tensor
    """
    if pos + 1 <= max_ctxt_len:
        # Simple case: just truncate
        return cache_tensor[:, :, : pos + 1, :]
    else:
        # Complex case: attention sink + recent context
        attention_sink = cache_tensor[:, :, 0:1, :]  # [batch, heads, 1, head_dim]
        recent_start = pos + 1 - (max_ctxt_len - 1)
        recent_context = cache_tensor[
            :, :, recent_start : pos + 1, :
        ]  # [batch, heads, max_ctxt_len-1, head_dim]
        return torch.cat([attention_sink, recent_context], dim=2)


def get_sliding_window_cache(
    full_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], pos: int, max_ctxt_len: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Get sliding window cache using optree.tree_map.

    Args:
        full_kv_cache: Legacy cache format [(keys, values), ...] for each layer
        pos: Current position
        max_ctxt_len: Maximum context length

    Returns:
        windowed_cache: Legacy cache format with sliding window applied
    """

    def slice_fn(tensor):
        return slice_cache_tensor_for_sliding_window(tensor, pos, max_ctxt_len)

    return optree.tree_map(slice_fn, full_kv_cache)


def expand_cache_for_batch(
    cache_legacy: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
) -> DynamicCache:
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


def sample_next_tokens(
    model,
    full_hidden: torch.Tensor,
    num_samples: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Sample next tokens for all positions in the sequence at once.

    Args:
        model: Upper layers model with calculate_logits method
        full_hidden: [seq_len, hidden_dim] - full sequence hidden states
        num_samples: Number of samples to draw per position
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run sampling on

    Returns:
        sampled_token_ids: [num_samples, seq_len] - sampled next tokens for each position
    """
    with torch.no_grad():
        seq_len, hidden_dim = full_hidden.shape

        # Move to device and add batch dimension
        hidden_batch = full_hidden.unsqueeze(0).to(device)  # [1, seq_len, hidden_dim]

        # Get logits for all positions at once
        logits = model.calculate_logits(hidden_batch)  # [1, seq_len, vocab_size]
        logits = logits.squeeze(0) / temperature  # [seq_len, vocab_size]

        # Apply nucleus sampling if specified
        if top_p < 1.0:
            # Process each position separately for nucleus sampling
            for pos in range(seq_len):
                pos_logits = logits[pos]  # [vocab_size]

                sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[pos, indices_to_remove] = float("-inf")

        # Sample multiple tokens for each position
        probs = torch.softmax(logits, dim=-1)  # [seq_len, vocab_size]

        # Sample num_samples tokens for each position
        sampled_indices = torch.multinomial(
            probs, num_samples, replacement=True
        ).T  # [num_samples, seq_len]

    return sampled_indices.cpu()


def load_sequences(
    tokenizer, max_seq_len: int = 512, num_sequences: int = 1000
) -> List[Tuple[torch.Tensor, str, str]]:
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
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )

    # Ensure EOS token exists
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    # For GPT-2, use EOS as attention sink at beginning
    attention_sink_id = tokenizer.eos_token_id

    sequences = []
    processed = 0

    for example in dataset:
        if len(sequences) >= num_sequences:
            break

        text = example["text"]
        title = example.get("title", "Unknown")

        if len(text.strip()) < 100:  # Skip very short texts
            continue

        # Tokenize with room for attention sink at beginning
        tokens = tokenizer(
            text,
            max_length=max_seq_len - 1,  # Leave room for attention sink
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Prepend attention sink (EOS token)
        tokens_with_sink = torch.cat([torch.tensor([attention_sink_id]), tokens])

        # Skip sequences that are too short for meaningful analysis
        if len(tokens_with_sink) < 20:
            continue

        # Create text preview for metadata
        text_preview = text[:200].replace("\n", " ").strip()

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
    sampled_token_ids: torch.Tensor,
    max_ctxt_len: int,
    num_samples: int,
    jacrev_chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    """
    Calculate Fisher matrix for a single position.

    Returns:
        fisher_matrix: [hidden_dim, hidden_dim]
        metadata: dict with position info
        sampled_tokens: [num_samples]
    """
    # Get sliding window context for this position
    context_hidden = get_sliding_window_hidden(full_hidden, pos, max_ctxt_len)

    # Get sliding window cache for this position
    context_cache_legacy = get_sliding_window_cache(full_kv_cache, pos, max_ctxt_len)

    # Move to device and expand for batch processing
    context_cache_on_device = optree.tree_map(
        lambda x: x.to(device), context_cache_legacy
    )
    batch_context_cache = expand_cache_for_batch(context_cache_on_device, num_samples)

    # Process each sampled token with the context cache
    final_latents_batch = gpt_lower.forward(
        sampled_token_ids.unsqueeze(-1).to(device),  # [num_samples, 1]
        past_key_value=batch_context_cache,
        use_cache=True,
    )[:, -1, :]  # [num_samples, hidden_dim] - just the new tokens

    # Use the sliding window context for upper layers too
    context_for_upper = (
        context_hidden.unsqueeze(0).expand(num_samples, -1, -1).contiguous().cuda()
    )

    # Compute gradients and probabilities
    grads, probs = gpt_upper.jacobian(
        final_latents_batch, context_for_upper, jacrev_chunk_size
    )

    # Compute Fisher matrices for all samples
    fisher_matrices_samples = fim_expected_gradient_outerproduct(grads, probs)

    # Average over samples
    fisher_avg = fisher_matrices_samples.mean(dim=0).cpu()  # [hidden_dim, hidden_dim]

    # Store metadata
    effective_context_len = len(context_hidden)
    metadata = {
        "position": pos,
        "effective_context_len": effective_context_len,
        "uses_sliding_window": pos + 1 > max_ctxt_len,
        "num_samples_used": len(sampled_token_ids),
        "sample_token_ids": sampled_token_ids.tolist(),
    }

    # All GPU tensors will be automatically cleaned up when this function returns
    return fisher_avg, metadata, sampled_token_ids


def calculate_fisher_for_single_sequence(
    lower_model,
    upper_model,
    sequence_tokens: torch.Tensor,
    max_ctxt_len: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_positions: int = None,
    position_skip: int = 1,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, List[dict], List[torch.Tensor]]:
    seq_len = len(sequence_tokens)
    max_pos = min(seq_len - 1, max_positions) if max_positions else seq_len - 1

    # 1. Process full sequence once to get hidden states and cache
    full_hidden, full_kv_cache = lower_model.forward_with_cache(
        sequence_tokens.unsqueeze(0).to(device)
    )
    full_hidden = full_hidden.squeeze(0).cpu()  # [seq_len, hidden_dim]
    full_kv_cache = optree.tree_map(lambda x: x.cpu(), full_kv_cache)
    hidden_dim = full_hidden.shape[-1]

    # 2. Sample tokens to use for Fisher-Stein calculation
    sampled_token_ids = sample_next_tokens(
        upper_model, full_hidden, num_samples, temperature, top_p
    )

    # 2. Determine which positions to process
    positions_to_process = list(range(position_skip - 1, max_pos, position_skip))

    # 2. Initialize results
    fisher_matrices = torch.zeros(max_pos, hidden_dim, hidden_dim, dtype=torch.float32)
    position_metadata = []
    sampled_tokens_list = []

    # 3. Process selected positions only
    for pos in tqdm(positions_to_process, desc="Processing positions"):
        fisher_matrix, metadata, sampled_tokens = calculate_fisher_at_position(
            pos,
            lower_model,
            upper_model,
            full_hidden,
            full_kv_cache,
            sampled_token_ids[:, pos],
            max_ctxt_len,
            num_samples,
            jacrev_chunk_size,
            device,
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
    max_ctxt_len: int,
    num_samples: int,
    jacrev_chunk_size: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_positions: Optional[int] = None,
    position_skip: int = 1,
    start_sequence_idx: int = 0,
    s3_bucket: Optional[str] = None,
    s3_key_prefix: str = "fisher-stein-results/",
    device: str = "cuda:0",
) -> None:
    """
    Process all sequences and save results.
    """

    lower_model = LowerLayersModel(model_name, layer_idx, device)
    upper_model = UpperLayersModel(model_name, layer_idx, device)

    for seq_idx, (sequence_tokens, text_preview, title) in tqdm(
        enumerate(sequences[start_sequence_idx:]), desc="Processing sequences"
    ):
        actual_seq_idx = start_sequence_idx + seq_idx
        logging.info(
            f"Processing sequence {actual_seq_idx + 1}/{len(sequences)}: {title}"
        )

        # Calculate Fisher matrices for this sequence
        fisher_matrices, position_metadata, sampled_tokens = (
            calculate_fisher_for_single_sequence(
                lower_model,
                upper_model,
                sequence_tokens,
                max_ctxt_len,
                num_samples,
                jacrev_chunk_size,
                temperature,
                top_p,
                max_positions,
                position_skip,
                device,
            )
        )

        # Create sequence metadata
        sequence_metadata = {
            "sequence_id": actual_seq_idx,
            "model_name": model_name,
            "layer_idx": layer_idx,
            "title": title,
            "text_preview": text_preview,
            "sequence_length": len(sequence_tokens),
            "max_ctxt_len": max_ctxt_len,
            "num_samples": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "position_skip": position_skip,
            "positions_analyzed": len(position_metadata),
            "full_sequence_tokens": sequence_tokens.tolist(),
            "position_metadata": position_metadata,
        }

        # Save results
        filename = f"fisher_seq_{actual_seq_idx:06d}.npz"
        save_results(
            fisher_matrices,
            sampled_tokens,
            sequence_metadata,
            filename,
            output_dir,
            s3_bucket,
            s3_key_prefix,
        )

        logging.info(f"Processed sequence {actual_seq_idx + 1}")


def main():
    args = parser.parse_args()

    setup_logging()

    # Validate arguments
    if args.layer_idx < 0:
        raise ValueError("layer_idx must be >= 0")

    if args.max_ctxt_len > args.max_seq_len:
        raise ValueError("max_ctxt_len cannot be greater than max_seq_len")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "num_samples": args.num_samples,
        "max_seq_len": args.max_seq_len,
        "max_ctxt_len": args.max_ctxt_len,
        "jacrev_chunk_size": args.jacrev_chunk_size,
        "num_sequences": args.num_sequences,
        "max_positions": args.max_positions,
        "position_skip": args.position_skip,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "start_sequence_idx": args.start_sequence_idx,
        "s3_bucket": args.s3_bucket,
        "s3_key_prefix": args.s3_key_prefix,
        "device": args.device,
    }

    save_json_results(
        config, "config.json", output_dir, args.s3_bucket, args.s3_key_prefix
    )

    logging.info(f"Configuration: {config}")
    logging.info(
        f"Processing sequences one at a time with {args.num_samples} samples per position"
    )
    logging.info(f"Using sliding window context with max length: {args.max_ctxt_len}")

    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load sequences
    sequences = load_sequences(
        tokenizer, max_seq_len=args.max_seq_len, num_sequences=args.num_sequences
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
        args.max_ctxt_len,
        args.num_samples,
        args.jacrev_chunk_size,
        args.temperature,
        args.top_p,
        args.max_positions,
        args.position_skip,
        args.start_sequence_idx,
        args.s3_bucket,
        args.s3_key_prefix,
        args.device,
    )

    logging.info("Completed Fisher matrix calculation!")

    # Save summary statistics
    summary = {
        "total_sequences_processed": len(sequences),
        "sequences_analyzed": f"{args.start_sequence_idx} to {min(args.start_sequence_idx + len(sequences), args.num_sequences)}",
        "config": config,
    }

    save_json_results(
        summary, "summary.json", output_dir, args.s3_bucket, args.s3_key_prefix
    )
    logging.info("Summary saved")


if __name__ == "__main__":
    main()
