"""Motion-Aware Precipitation Nowcasting Architecture.

Paper Motivation:
  Catão et al., "Precipitation nowcasting of satellite data using physically aligned
  neural networks", Environmental Data Science, 2026.
  Shi et al., "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model", 2017.

Mechanism:
  Integrates explicit atmospheric storm-motion velocity vectors (u, v) derived from
  optical flow alongside satellite infrared and precipitation observations.
  Provides physical kinematic advection priors directly to the spatiotemporal encoder,
  testing whether explicit velocity fields improve trajectory preservation at +60m to +360m.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.experimental.simvp import BasicConv2d, GatedTemporalBlock


class MotionSimVPWeather(nn.Module):
    """Motion-Aware SimVP integrating satellite observations with optical flow advection fields.

    Args:
        in_channels: Number of input channels per frame (default: 5: IR, Precip, Mask, U_flow, V_flow).
        hidden_dim: Latent spatial channel dimension (default: 32).
        t_in: Number of historical input timesteps (default: 4).
        num_leads: Number of future forecast lead times (default: 5).
        num_translator_blocks: Depth of translator stack (default: 3).
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 32,
        t_in: int = 4,
        num_leads: int = 5,
        num_translator_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_in = t_in
        self.num_leads = num_leads
        self.hidden_dim = hidden_dim

        # 1. Motion-Conditioned Spatial Encoder
        self.encoder = nn.Sequential(
            BasicConv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            BasicConv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        )

        # 2. Spatiotemporal Translator
        trans_in_channels = t_in * hidden_dim
        trans_out_channels = num_leads * hidden_dim

        self.trans_in = BasicConv2d(trans_in_channels, trans_out_channels, kernel_size=1, padding=0)
        translator_layers = []
        for _ in range(num_translator_blocks):
            translator_layers.append(GatedTemporalBlock(trans_out_channels))
        self.translator = nn.Sequential(*translator_layers)

        # 3. Spatial Decoder: Multi-head prediction
        cat_heads = []
        qpf_heads = []
        for _ in range(num_leads):
            cat_heads.append(
                nn.Sequential(
                    BasicConv2d(hidden_dim, 24, kernel_size=3, padding=1),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(24, 1, kernel_size=1),
                )
            )
            qpf_heads.append(
                nn.Sequential(
                    BasicConv2d(hidden_dim, 24, kernel_size=3, padding=1),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(24, 1, kernel_size=1),
                    nn.Softplus(),
                )
            )
        self.cat_heads = nn.ModuleList(cat_heads)
        self.qpf_heads = nn.ModuleList(qpf_heads)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through motion-aware spatiotemporal pipeline.

        Args:
            x_seq: (B, T_in, 5, H, W) tensor where channels are [IR, Precip, Mask, U_flow, V_flow].

        Returns:
            Tuple of:
              - cat_logits: (B, num_leads, H, W) occurrence logits
              - qpf_rates: (B, num_leads, H, W) continuous rain rates (mm/hr)
        """
        B, T, C, H, W = x_seq.shape

        x_flat = x_seq.view(B * T, C, H, W)
        enc_feat = self.encoder(x_flat)

        enc_feat = enc_feat.view(B, T * self.hidden_dim, H, W)
        trans_feat = self.trans_in(enc_feat)
        trans_feat = self.translator(trans_feat)

        lead_feats = trans_feat.view(B, self.num_leads, self.hidden_dim, H, W)

        cat_outs = []
        qpf_outs = []
        for k in range(self.num_leads):
            h_k = lead_feats[:, k]
            cat_outs.append(self.cat_heads[k](h_k).squeeze(1))
            qpf_outs.append(self.qpf_heads[k](h_k).squeeze(1))

        return torch.stack(cat_outs, dim=1), torch.stack(qpf_outs, dim=1)
