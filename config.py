from __future__ import annotations
from typing import Any, Iterator
from copy import deepcopy

class Config:
    """A flexible configuration class that supports attribute-style access and updates."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize Config with optional keyword arguments.
        
        Args:
            **kwargs: Arbitrary keyword arguments to set as config values
        """
        for key, value in kwargs.items():
            setattr(self, key, deepcopy(value))
    
    def update(self, other: Config) -> None:
        """Update current config with values from another config.
        
        Args:
            other: Another Config instance whose values will be used to update this one
        """
        for key, value in other:
            setattr(self, key, deepcopy(value))
    
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Make Config iterable, yielding (key, value) pairs.
        
        Returns:
            Iterator of (key, value) pairs
        """
        return iter(vars(self).items())
    
    def __repr__(self) -> str:
        """Return a string representation of the Config object.
        
        Returns:
            A string showing the config contents
        """
        attrs = vars(self)
        items = [f"{k}={repr(v)}" for k, v in attrs.items()]
        return f"Config({', '.join(items)})"

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary notation.
        
        Args:
            key: The configuration key to retrieve
            
        Returns:
            The value associated with the key
            
        Raises:
            KeyError: If the key doesn't exist
        """
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dictionary notation.
        
        Args:
            key: The configuration key to set
            value: The value to associate with the key
        """
        setattr(self, key, deepcopy(value))

    def __len__(self) -> int:
        """Return the number of attributes in the Config object.
        
        Returns:
            The number of attributes in the Config object
        """
        return len(vars(self))

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the Config object.
        
        Args:
            key: The key to check for
            
        Returns:
            True if the key is in the Config object, False otherwise
        """
        return key in vars(self)

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the Config object.
        
        Returns:
            A dictionary representation of the Config object
        """
        return vars(self)
