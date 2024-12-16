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
