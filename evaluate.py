"""
Evaluation utilities for the model
"""

from __future__ import annotations

import timeit
import torch

from contextlib import nullcontext
from datetime import datetime

import torch
from tqdm import tqdm

from config import Config
torch.serialization.add_safe_globals([Config])
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
    float16_inference=False,  # whether to convert model to float16 for inference
    int8_quantization=False,  # whether to convert model to int8 for inference
)


def compute_perplexity(model: GPT, config: Config, ctx: nullcontext | torch.amp.autocast) -> tuple[float, float]:
    """Compute perplexity of the model"""
    X, y = get_validation(config)

    # Calculate number of batches
    n_samples = X.shape[0]
    n_batches = (n_samples + config.batch_size - 1) // config.batch_size

    total_log_probs = []

    start_time = timeit.default_timer()
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
    end_time = timeit.default_timer()
    tokens_per_second = n_samples / (end_time - start_time)

    return perplexity, tokens_per_second


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    global config

    config = get_config_from_args(config=config)  # overrides from command line or config file

    device_type, ptdtype, ctx = setup_torch_config(config)
    model = load_model(config)
    
    if config.float16_inference:
        print("Converting model to float16 for inference")
        model = model.half()  # Convert to float16

    if config.int8_quantization:
        print("Converting model to int8 for inference (dynamic quantization)")
        layers = {torch.nn.Linear}
        torch.backends.quantized.engine = 'qnnpack'
        torch.ao.quantization.quantize_dynamic(model, layers, dtype=torch.qint8, inplace=True)
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    start_time = timeit.default_timer()
    perplexity, tokens_per_second = compute_perplexity(model, config, ctx)
    end_time = timeit.default_timer()
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Perplexity: {perplexity:.2f}")

    d = {
        "model_params": model.get_num_params(),
        "config": config.__dict__,
        "perplexity": perplexity,
        "tokens_per_second": tokens_per_second,
        "total_time": end_time - start_time,
        "model_size": size_all_mb,
    }
    import json
    import os

    os.makedirs("eval", exist_ok=True)
    with open(f"eval/evaluate_{timestamp}.json", "w") as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
