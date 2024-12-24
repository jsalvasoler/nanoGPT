"""
Sample from a trained model
"""

from __future__ import annotations

import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from config import Config
from configurator import get_config_from_args
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Default configuration - these values can be overridden by configurator.py
config = Config(
    # model initialization
    init_from="resume",  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir="out",  # ignored if init_from is not 'resume'
    start="\n",  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    # sampling parameters
    num_samples=10,  # number of samples to draw
    max_new_tokens=500,  # number of tokens generated in each sample
    temperature=0.8,  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k=200,  # retain only the top_k most likely tokens, clamp others to have 0 probability
    # system parameters
    seed=1337,
    device="cuda",  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    compile=False,  # use PyTorch 2.0 to compile the model to be faster
    perplexity=True,
)
# -----------------------------------------------------------------------------


def setup_torch_config(config: Config) -> tuple[str, torch.dtype, nullcontext | torch.amp.autocast]:
    """Set up PyTorch configuration and return device type, dtype and context"""
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in config.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    return device_type, ptdtype, ctx


def load_model(config: Config) -> GPT:
    """Load and configure the model"""
    if config.init_from == "resume":
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif config.init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(config.init_from, dict(dropout=0.0))

    model.eval()
    model.to(config.device)
    if config.compile:
        model = torch.compile(model)
    return model


def setup_encoding(init_from: str, checkpoint: dict | None = None) -> tuple[callable, callable]:
    """Set up encoding and decoding functions"""
    load_meta = False
    if init_from == "resume" and checkpoint and "config" in checkpoint and "dataset" in checkpoint["config"]:
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


def generate_samples(
    model: GPT, encode: callable, decode: callable, config: Config, ctx: nullcontext | torch.amp.autocast
) -> None:
    """Generate and print samples from the model"""
    # encode the beginning of the prompt
    if config.start.startswith("FILE:"):
        with open(config.start[5:], encoding="utf-8") as f:
            start = f.read()
    else:
        start = config.start
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(config.num_samples):
                y = model.generate(x, config.max_new_tokens, temperature=config.temperature, top_k=config.top_k)
                print(decode(y[0].tolist()))
                print("---------------")


def main() -> None:
    global config

    # Load config overrides
    config = get_config_from_args(config=config)  # overrides from command line or config file

    device_type, ptdtype, ctx = setup_torch_config(config)
    model = load_model(config)

    # Get the checkpoint if we're resuming
    checkpoint = None
    if config.init_from == "resume":
        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)

    encode, decode = setup_encoding(config.init_from, checkpoint)

    generate_samples(model, encode, decode, config, ctx)


if __name__ == "__main__":
    main()
