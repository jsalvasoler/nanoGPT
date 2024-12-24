"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

from __future__ import annotations

import math
import os
import pickle
import time
from contextlib import nullcontext
from typing import Any

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from configurator import get_config_from_args
from model import GPT, GPTConfig
from utils import get_batch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
config = Config(
    # I/O
    out_dir="out",
    eval_interval=2000,
    log_interval=1,
    eval_iters=200,
    eval_only=False,  # if True, script exits right after the first eval
    always_save_checkpoint=True,  # if True, always save a checkpoint after each eval
    init_from="scratch",  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log=False,  # disabled by default
    wandb_project="owt",
    wandb_run_name="gpt2",  # 'run' + str(time.time())
    # data
    dataset="openwebtext",
    gradient_accumulation_steps=5 * 8,  # used to simulate larger batch sizes
    batch_size=12,  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size=1024,
    # model
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,  # for pretraining 0 is good, for finetuning try 0.1+
    bias=False,  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate=6e-4,  # max learning rate
    max_iters=600000,  # total number of training iterations
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr=True,  # whether to decay the learning rate
    warmup_iters=2000,  # how many steps to warm up for
    lr_decay_iters=600000,  # should be ~= max_iters per Chinchilla
    min_lr=6e-5,  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend="nccl",  # 'nccl', 'gloo', etc.
    # system
    device="cuda",  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype=("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"),
    compile=True,  # use PyTorch 2.0 to compile the model to be faster
)
# -----------------------------------------------------------------------------


def setup_training(config: Config) -> tuple[dict[str, Any], GPT, torch.optim.Optimizer, torch.amp.GradScaler]:
    """Setup training environment and initialize model, optimizer etc."""
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        config.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config.device

    # setup state dict to return
    training_state = {
        "ddp": ddp,
        "master_process": master_process,
        "seed_offset": seed_offset,
        "ddp_world_size": ddp_world_size,
        "tokens_per_iter": config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size,
        "best_val_loss": 1e9,
        "iter_num": 0,
    }

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize model
    model, optimizer = setup_model(config, device)

    # Initialize grad scaler
    scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"))

    if config.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return training_state, model, optimizer, scaler


def setup_model(config: Config, device: str) -> tuple[GPT, torch.optim.Optimizer]:
    """Initialize and setup the model and optimizer."""
    # attempt to derive vocab_size from the dataset
    meta_vocab_size = get_vocab_size(config)

    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=None,
        dropout=config.dropout,
    )

    model, model_args = initialize_model(config, model_args, meta_vocab_size, device)
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, (config.beta1, config.beta2), "cuda" if "cuda" in device else "cpu"
    )

    return model, optimizer


def get_vocab_size(config: Config) -> int | None:
    """Get vocabulary size from dataset metadata."""
    data_dir = os.path.join("data", config.dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        return meta_vocab_size
    return None


def initialize_model(config: Config, model_args: dict, meta_vocab_size: int | None, device: str) -> tuple[GPT, dict]:
    """Initialize model based on config settings."""
    if config.init_from == "scratch":
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config.init_from == "resume":
        model, model_args = resume_from_checkpoint(config, model_args, device)
    elif config.init_from.startswith("gpt2"):
        model, model_args = initialize_from_gpt2(config, model_args)

    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args["block_size"] = config.block_size

    model.to(device)
    return model, model_args


def resume_from_checkpoint(config: Config, model_args: dict, device: str) -> tuple[GPT, dict]:
    """Resume training from a checkpoint."""
    print(f"Resuming training from {config.out_dir}")
    ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    # force these config attributes to be equal otherwise we can't resume training
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    # fix the keys of the state dictionary
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    return model, model_args


def initialize_from_gpt2(config: Config, model_args: dict) -> tuple[GPT, dict]:
    """Initialize from OpenAI GPT-2 weights."""
    print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)

    # read off the created config params
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

    return model, model_args


@torch.no_grad()
def estimate_loss(model: GPT, config: Config, ctx: nullcontext) -> dict[str, float]:
    """Estimate loss over train and validation splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config, "cuda" if "cuda" in config.device else "cpu")
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(iter_num: int, config: Config) -> float:
    """Learning rate decay scheduler (cosine with warmup)."""
    # 1) linear warmup for warmup_iters steps
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)

    # 2) if iter > lr_decay_iters, return min learning rate
    if iter_num > config.lr_decay_iters:
        return config.min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train_loop(
    config: Config, model: GPT, optimizer: torch.optim.Optimizer, scaler: torch.cuda.GradScaler, training_state: dict
) -> None:
    """Main training loop."""
    # setup context
    device_type = "cuda" if "cuda" in config.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init wandb if needed
    if config.wandb_log and training_state["master_process"]:
        import wandb

        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.as_dict())

    # training loop
    X, Y = get_batch("train", config, device_type)
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if training_state["ddp"] else model
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(training_state["iter_num"], config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if training_state["iter_num"] % config.eval_interval == 0 and training_state["master_process"]:
            losses = estimate_loss(model, config, ctx)
            print(f"step {training_state['iter_num']}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if config.wandb_log:
                wandb.log(
                    {
                        "iter": training_state["iter_num"],
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }
                )

            if losses["val"] < training_state["best_val_loss"] or config.always_save_checkpoint:
                training_state["best_val_loss"] = losses["val"]
                if training_state["iter_num"] > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model.config.__dict__,
                        "iter_num": training_state["iter_num"],
                        "best_val_loss": training_state["best_val_loss"],
                        "config": config,
                    }
                    print(f"saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))

        if training_state["iter_num"] == 0 and config.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(config.gradient_accumulation_steps):
            if training_state["ddp"]:
                # in DDP training we only need to sync gradients at the last micro step
                model.require_backward_grad_sync = micro_step == config.gradient_accumulation_steps - 1

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.gradient_accumulation_steps

            X, Y = get_batch("train", config, device_type)
            scaler.scale(loss).backward()

        # clip the gradient
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if training_state["iter_num"] % config.log_interval == 0 and training_state["master_process"]:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {training_state['iter_num']}: "
                f"loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                f"mfu {running_mfu*100:.2f}%"
            )

        training_state["iter_num"] += 1
        local_iter_num += 1

        # termination conditions
        if training_state["iter_num"] > config.max_iters:
            break


def main() -> None:
    """Main function to setup and start training."""
    # overrides from command line or config file
    global config
    config = get_config_from_args(config=config)

    # Setup training environment
    training_state, model, optimizer, scaler = setup_training(config)

    # Start training
    train_loop(config, model, optimizer, scaler, training_state)

    if training_state["ddp"]:
        destroy_process_group()


if __name__ == "__main__":
    main()
