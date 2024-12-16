from __future__ import annotations
import pytest
from config import Config
from configurator import get_config, get_config_from_args


def test_get_config_known_config() -> None:
    """Test getting a known config."""
    config = get_config("eval_gpt2")
    assert isinstance(config, Config)


def test_get_config_unknown_config() -> None:
    """Test getting an unknown config raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config name: unknown_config"):
        get_config("unknown_config")


def test_get_config_from_args_empty() -> None:
    """Test processing empty args list."""
    config = get_config_from_args([])
    assert isinstance(config, Config)
    assert len(config) == 0


def test_get_config_from_args_file() -> None:
    """Test processing a config file."""
    config = get_config_from_args(["finetune_shakespeare"])
    assert isinstance(config, Config)

    assert config.dataset == "shakespeare"

    config_py = get_config_from_args(["finetune_shakespeare.py"])
    config_config_py = get_config_from_args(["config/finetune_shakespeare.py"])
    for k, v in config:
        assert config_py[k] == v
        assert config_config_py[k] == v


def test_update_parameters_from_shakespeare() -> None:
    """Test updating parameters from a shakespeare config."""
    config = get_config_from_args(["finetune_shakespeare", "--dataset=x"])
    assert config.dataset == "x"


def test_update_non_existent_parameter() -> None:
    """Test updating a non-existent parameter."""
    with pytest.raises(ValueError, match="Unknown config key: doesnotexist"):
        get_config_from_args(["finetune_shakespeare", "--doesnotexist=x"])


def test_update_parameter_with_invalid_type() -> None:
    """Test updating a parameter with an invalid value."""
    with pytest.raises(AssertionError):
        get_config_from_args(["finetune_shakespeare", "--batch_size=hello"])


def test_get_config_from_args_starting_with_config() -> None:
    config = Config(a=1)
    config_from_args = get_config_from_args(args=["--a=2"], config=config)
    assert config_from_args.a == 2

    with pytest.raises(ValueError, match="Unknown config key: b"):
        get_config_from_args(args=["--b=3"], config=config)