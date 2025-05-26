"""Indexing schemas v5 d285."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class IndexingConfig5D285:
    enabled: bool = True
    batch_size: int = 160
    hidden_dim: int = 320
    num_layers: int = 7
    dropout: float = 0.5
    lr: float = 5.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndexingConfig5D285":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
