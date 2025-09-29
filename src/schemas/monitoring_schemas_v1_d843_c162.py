"""Monitoring schemas v1 d843."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class MonitoringConfig1D843:
    enabled: bool = True
    batch_size: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MonitoringConfig1D843":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
