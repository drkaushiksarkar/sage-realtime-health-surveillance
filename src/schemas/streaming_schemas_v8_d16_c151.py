"""Streaming schemas v8 d16."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class StreamingConfig8D16:
    enabled: bool = True
    batch_size: int = 256
    hidden_dim: int = 512
    num_layers: int = 10
    dropout: float = 0.8
    lr: float = 8.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamingConfig8D16":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
