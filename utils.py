import os

import numpy as np
import torch

from config import Config


def get_batch(split: str, config: Config, device_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data."""
    data_dir = os.path.join("data", config.dataset)

    # Recreate np.memmap every batch to avoid memory leak
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + config.block_size]).astype(np.int64)) for i in ix])

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)

    return x, y


def get_validation(config: Config, calibration: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Get all possible , return in a batch of size 1"""
    data_dir = os.path.join("data", config.dataset)
    name = "val.bin" if not calibration else "cal.bin"
    data = np.memmap(os.path.join(data_dir, name), dtype=np.uint16, mode="r")

    ix = torch.arange(len(data) - config.block_size)
    x = torch.stack([torch.from_numpy((data[i : i + config.block_size]).astype(np.int64)) for i in ix])
    y = torch.from_numpy(data[config.block_size :].copy().astype(np.int64))

    assert x.shape[0] == y.shape[0]

    if config.device == "cuda":
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)

    return x, y
