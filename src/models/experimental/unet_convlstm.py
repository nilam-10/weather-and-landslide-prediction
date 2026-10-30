"""Compact U-Net-ConvLSTM Architecture for Satellite Precipitation Nowcasting.

Paper Reference:
  Rahimi et al., "Global Precipitation Nowcasting of Integrated Multi-satellitE Retrievals
  for GPM: A U-Net Convolutional LSTM Architecture", Journal of Hydrometeorology, 2024.
  DOI: 10.1175/JHM-D-23-0119.1

Key Architectural Mechanism:
  Multi-scale spatial downsampling with ConvLSTM recurrent memory at each hierarchy:
    1. Encoder Level 1 (Full Resolution: 25 x 40): Captures fine convective precipitation cores.
    2. Encoder Level 2 (Downsampled: 12 x 20): Expands spatial receptive field to capture
       synoptic-scale advection and cloud systems.
    3. Decoder with Skip Connections: Fuses multi-scale spatiotemporal features and restores
       fine grid resolution via bilinear upsampling and skip concatenation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.convlstm import ConvLSTMCell


class UNetConvLSTMWeather(nn.Module):
    """Compact Multi-Scale U-Net-ConvLSTM for Satellite Weather Nowcasting.

    Args:
        in_channels: Number of input channels (default: 3).
        base_channels: Number of base feature channels at Level 1 (default: 24).
        num_leads: Number of forecast lead times (default: 5).
        dropout: Spatial dropout rate (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        num_leads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_leads = num_leads
        c1 = base_channels
        c2 = base_channels * 2  # 48

        # --- Encoder Level 1 (Resolution: 25 x 40) ---
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1_cell = ConvLSTMCell(in_channels=c1, hidden_channels=c1, kernel_size=3)

        # --- Encoder Level 2 (Resolution: 12 x 20 via 2x2 MaxPool) ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2_cell = ConvLSTMCell(in_channels=c2, hidden_channels=c2, kernel_size=3)

        # --- Decoder & Multi-Scale Feature Fusion ---
        # Level 2 upsampled to Level 1 and concatenated with Level 1 skip connection
        self.dec_fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # --- Multi-Head Lead Projections ---
        cat_heads = []
        qpf_heads = []
        for _ in range(num_leads):
            cat_heads.append(
                nn.Sequential(
                    nn.Conv2d(c1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(16, 1, kernel_size=1),
                )
            )
            qpf_heads.append(
                nn.Sequential(
                    nn.Conv2d(c1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(16, 1, kernel_size=1),
                    nn.Softplus(),
                )
            )
        self.cat_heads = nn.ModuleList(cat_heads)
        self.qpf_heads = nn.ModuleList(qpf_heads)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through U-Net-ConvLSTM pipeline.

        Args:
            x_seq: (B, T, C, H, W) input tensor.

        Returns:
            Tuple of:
              - cat_logits: (B, num_leads, H, W) occurrence logits
              - qpf_rates: (B, num_leads, H, W) continuous rain rates (mm/hr)
        """
        B, T, C, H, W = x_seq.shape

        state1 = None
        state2 = None

        for t in range(T):
            x_t = x_seq[:, t]

            # Level 1 Recurrent Step (25 x 40)
            feat1 = self.enc1_conv(x_t)
            h1, c1 = self.enc1_cell(feat1, state1)
            state1 = (h1, c1)

            # Level 2 Recurrent Step (12 x 20)
            pooled = self.pool(h1)
            feat2 = self.enc2_conv(pooled)
            h2, c2 = self.enc2_cell(feat2, state2)
            state2 = (h2, c2)

        # Final time step states
        last_h1 = state1[0]  # (B, 24, 25, 40)
        last_h2 = state2[0]  # (B, 48, 12, 20)

        # Upsample Level 2 back to Level 1 resolution (25 x 40)
        h2_up = F.interpolate(last_h2, size=(H, W), mode="bilinear", align_corners=False)

        # Fuse skip connection from Level 1 with upsampled Level 2 features
        fused = self.dec_fuse(torch.cat([last_h1, h2_up], dim=1))  # (B, 24, 25, 40)

        # Multi-lead head outputs
        cat_outs = [head(fused).squeeze(1) for head in self.cat_heads]
        qpf_outs = [head(fused).squeeze(1) for head in self.qpf_heads]

        return torch.stack(cat_outs, dim=1), torch.stack(qpf_outs, dim=1)
