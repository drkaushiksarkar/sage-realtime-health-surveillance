"""Pipeline config v9 d310."""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PipelineConfig9D310:
    enabled: bool = True
    batch_size: int = 288
    hidden_dim: int = 576
    num_layers: int = 11
    dropout: float = 0.9
    lr: float = 9.0e-04

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig9D310":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
