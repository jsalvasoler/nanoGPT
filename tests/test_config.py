from __future__ import annotations
import pytest
from config import Config


def test_config_init() -> None:
    """Test Config initialization with kwargs."""
    config = Config(host="localhost", port=8000)
    assert config.host == "localhost"
    assert config.port == 8000


def test_config_set_attr() -> None:
    """Test setting attributes after initialization."""
    config = Config()
    config.debug = True
    config.api_key = "secret123"
    
    assert config.debug is True
    assert config.api_key == "secret123"


def test_config_missing_attr() -> None:
    """Test accessing non-existent attributes raises AttributeError."""
    config = Config()
    with pytest.raises(AttributeError):
        _ = config.nonexistent


def test_config_repr() -> None:
    """Test string representation of Config."""
    config = Config(name="test", value=42)
    assert repr(config) == "Config(name='test', value=42)"


def test_config_multiple_updates() -> None:
    """Test multiple updates to the same attribute."""
    config = Config()
    config.counter = 1
    config.counter = 2
    assert config.counter == 2


def test_config_different_value_types() -> None:
    """Test Config can handle different value types."""
    config = Config()
    
    # Test various Python types
    config.string = "hello"
    config.integer = 42
    config.float_num = 3.14
    config.boolean = True
    config.none_value = None
    config.list_value = [1, 2, 3]
    config.dict_value = {"key": "value"}
    
    assert config.string == "hello"
    assert config.integer == 42
    assert config.float_num == 3.14
    assert config.boolean is True
    assert config.none_value is None
    assert config.list_value == [1, 2, 3]
    assert config.dict_value == {"key": "value"}


def test_config_update() -> None:
    """Test updating config with another config."""
    config1 = Config(host="localhost", port=8000)
    config2 = Config(port=9000, debug=True)
    
    config1.update(config2)
    assert config1.host == "localhost"  # Keeps original value
    assert config1.port == 9000        # Updated value
    assert config1.debug is True       # New value


def test_config_iteration() -> None:
    """Test that Config can be iterated over."""
    config = Config(host="localhost", port=8000, debug=True)
    
    # Convert to dict for easy comparison
    config_dict = dict(config)
    assert config_dict == {
        "host": "localhost",
        "port": 8000,
        "debug": True
    }
    
    # Test iteration directly
    keys = []
    values = []
    for key, value in config:
        keys.append(key)
        values.append(value)
    
    assert sorted(keys) == sorted(["host", "port", "debug"])
    assert set(values) == {"localhost", 8000, True}


def test_config_update_empty() -> None:
    """Test updating an empty config."""
    config1 = Config()
    config2 = Config(host="localhost", port=8000)
    
    config1.update(config2)
    assert config1.host == "localhost"
    assert config1.port == 8000


def test_config_update_nested_objects() -> None:
    """Test that update creates new references for nested objects."""
    config1 = Config(data={"a": 1})
    config2 = Config(data={"b": 2})
    
    config1.update(config2)
    assert config1.data == {"b": 2}  # Complete override
    
    # Verify it's a new reference
    config2.data["b"] = 3
    assert config1.data == {"b": 2}  # Original remains unchanged


def test_config_update_multiple() -> None:
    """Test multiple updates in sequence."""
    config1 = Config(a=1, b=2)
    config2 = Config(b=3, c=4)
    config3 = Config(c=5, d=6)
    
    config1.update(config2)
    config1.update(config3)
    
    assert config1.a == 1  # Original value preserved
    assert config1.b == 3  # Updated by config2
    assert config1.c == 5  # Updated by config3
    assert config1.d == 6  # Added by config3
