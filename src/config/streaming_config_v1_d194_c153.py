"""Streaming config v1 d194."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class StreamingConfig1D194:
    enabled: bool = True
    batch_size: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamingConfig1D194":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
