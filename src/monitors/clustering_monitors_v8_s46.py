"""ClusteringMonitorsV8S46: clustering_monitors_v8_s46 module v8.

Integrates with SAGE Data Lake (1.78B rows, 40K+ indicators, 85 organizations).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ClusteringMonitorsV8S46Config:
    """Configuration for clustering_monitors_v8_s46 v8."""
    batch_size: int = 256
    hidden_dim: int = 1024
    num_layers: int = 10
    dropout: float = 0.8
    learning_rate: float = 8.0e-04
    max_epochs: int = 200
    sage_table: str = "sage_lake.observations"
    device: str = "cuda"
    seed: int = 50


class ClusteringMonitorsV8S46(nn.Module):
    """Neural network module for clustering_monitors_v8_s46 v8.

    Architecture:
        Input -> LayerNorm -> N x (Linear -> GELU -> Dropout) -> Output
        With residual connections every 2 layers.
    """

    def __init__(self, config: Optional[ClusteringMonitorsV8S46Config] = None):
        super().__init__()
        self.config = config or ClusteringMonitorsV8S46Config()
        c = self.config

        self.input_proj = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.input_norm = nn.LayerNorm(c.hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(c.num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(c.hidden_dim, c.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.hidden_dim * 4, c.hidden_dim),
                nn.Dropout(c.dropout),
            ))

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(c.hidden_dim) for _ in range(c.num_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, 1),
        )

        self._step_count = 0
        self._metrics: Dict[str, List[float]] = {"loss": [], "grad_norm": []}
        logger.info("Initialized ClusteringMonitorsV8S46 v8 with %d params",
                     sum(p.numel() for p in self.parameters()))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with residual connections.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)

        Returns:
            Dict with predictions and intermediate representations
        """
        h = self.input_norm(self.input_proj(x))

        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = h
            h = layer(h)
            h = norm(h + residual)  # Pre-norm residual connection

        # Pool over sequence dimension
        pooled = h.mean(dim=1) if h.dim() == 3 else h
        output = self.output_head(pooled)

        return {
            "predictions": output.squeeze(-1),
            "representation": pooled,
            "hidden_states": h,
        }

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional label smoothing."""
        return nn.functional.mse_loss(predictions, targets)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs["predictions"], targets)

        self._step_count += 1
        self._metrics["loss"].append(loss.item())

        return {"loss": loss.item(), "step": self._step_count}

    def get_metrics(self) -> Dict[str, float]:
        return {
            k: np.mean(v[-100:]) if v else 0.0
            for k, v in self._metrics.items()
        }
