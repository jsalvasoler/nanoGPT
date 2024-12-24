"""
Evaluation utilities for the model
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from config import Config
from configurator import get_config_from_args
from model import GPT
from sample import load_model, setup_torch_config
from utils import get_validation

# -----------------------------------------------------------------------------
# Default configuration - these values can be overridden by configurator.py
config = Config(
    out_dir="out",
    init_from="resume",
    # model parameters for perplexity
    dataset="shakespeare_char",
    block_size=64,
    batch_size=12,
    # system parameters
    seed=1337,
    device="cuda",  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    compile=False,  # use PyTorch 2.0 to compile the model to be faster
)


def compute_perplexity(model: GPT, config: Config, ctx: nullcontext | torch.amp.autocast) -> float:
    """Compute perplexity of the model"""
    X, y = get_validation(config)

    # Calculate number of batches
    n_samples = X.shape[0]
    n_batches = (n_samples + config.batch_size - 1) // config.batch_size

    total_log_probs = []

    with torch.no_grad():
        with ctx:
            for i in tqdm(range(n_batches)):
                # Get batch indices
                start_idx = i * config.batch_size
                end_idx = min((i + 1) * config.batch_size, n_samples)

                # Get batch data
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Forward pass
                logits, _ = model(X_batch)
                logits = logits[:, -1, :]
                # convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                # get the predicted logits for the y index
                pred_probs = probs[torch.arange(len(y_batch)), y_batch.int()]
                # apply log
                log_probs = torch.log(pred_probs)
                total_log_probs.append(log_probs)

    # Concatenate all log probabilities and compute mean
    all_log_probs = torch.cat(total_log_probs)
    perplexity = torch.exp(-all_log_probs.mean()).item()

    return perplexity


def main() -> None:
    global config

    config = get_config_from_args(config=config)  # overrides from command line or config file

    device_type, ptdtype, ctx = setup_torch_config(config)
    model = load_model(config)

    perplexity = compute_perplexity(model, config, ctx)
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
