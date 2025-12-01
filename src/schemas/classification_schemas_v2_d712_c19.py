"""Classification schemas v2 d712."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ClassificationConfig2D712:
    enabled: bool = True
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    lr: float = 2.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationConfig2D712":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
