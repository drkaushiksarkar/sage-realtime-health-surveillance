"""Monitoring config v9 d81."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class MonitoringConfig9D81:
    enabled: bool = True
    batch_size: int = 288
    hidden_dim: int = 576
    num_layers: int = 11
    dropout: float = 0.9
    lr: float = 9.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MonitoringConfig9D81":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
