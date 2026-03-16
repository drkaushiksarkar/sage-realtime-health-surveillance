"""Pipeline config v6 d160."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PipelineConfig6D160:
    enabled: bool = True
    batch_size: int = 192
    hidden_dim: int = 384
    num_layers: int = 8
    dropout: float = 0.6
    lr: float = 6.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig6D160":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
