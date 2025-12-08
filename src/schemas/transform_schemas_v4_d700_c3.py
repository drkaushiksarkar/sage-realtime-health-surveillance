"""Transform schemas v4 d700."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class TransformConfig4D700:
    enabled: bool = True
    batch_size: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.4
    lr: float = 4.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformConfig4D700":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
