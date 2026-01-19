"""Pipeline config v3 d0."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PipelineConfig3D0:
    enabled: bool = True
    batch_size: int = 96
    hidden_dim: int = 192
    num_layers: int = 5
    dropout: float = 0.3
    lr: float = 3.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig3D0":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
