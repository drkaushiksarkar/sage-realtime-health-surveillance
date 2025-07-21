"""Clustering config v7 d142."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ClusteringConfig7D142:
    enabled: bool = True
    batch_size: int = 224
    hidden_dim: int = 448
    num_layers: int = 9
    dropout: float = 0.7
    lr: float = 7.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClusteringConfig7D142":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
