"""Sampling config v3 d957."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SamplingConfig3D957:
    enabled: bool = True
    batch_size: int = 96
    hidden_dim: int = 192
    num_layers: int = 5
    dropout: float = 0.3
    lr: float = 3.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig3D957":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
