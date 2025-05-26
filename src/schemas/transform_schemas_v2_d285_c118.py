"""Transform schemas v2 d285."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class TransformConfig2D285:
    enabled: bool = True
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    lr: float = 2.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformConfig2D285":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
