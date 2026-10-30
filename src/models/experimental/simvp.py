"""SimVP: Simpler Yet Better Video Prediction adapted for Satellite Weather Nowcasting.

Paper Reference:
  Gao et al., "SimVP: Simpler Yet Better Video Prediction", CVPR 2022.
  DOI: 10.1109/CVPR52688.2022.00317

Key Architectural Mechanism:
  Non-recurrent spatiotemporal predictive architecture:
    1. Spatial Encoder: 2D CNN layers extracting compact spatial representations.
    2. Spatiotemporal Translator: Stack of Gated Inception / Temporal Convolutional
       blocks mapping historical latent sequences to future latent states without
       sequential recurrent hidden state compounding.
    3. Spatial Decoder: Multi-head projection layers predicting multi-lead occurrence
       logits and continuous rain rates.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """Convolution + BatchNorm + LeakyReLU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GatedTemporalBlock(nn.Module):
    """Depthwise-separable gated spatiotemporal translation block with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        # Depthwise separable convolutions to keep parameters compact
        self.conv1_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv1_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv2_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.conv1_pw(self.conv1_dw(x))
        feat = F.leaky_relu(self.bn1(feat), 0.2, inplace=True)
        feat = self.conv2_pw(self.conv2_dw(feat))
        feat = self.bn2(feat)
        gate = torch.sigmoid(self.gate(x))
        out = feat * gate + residual
        return F.leaky_relu(out, 0.2, inplace=True)


class SimVPWeather(nn.Module):
    """SimVP architecture customized for multi-lead satellite precipitation nowcasting.

    Args:
        in_channels: Number of input channels per frame (default: 3).
        hidden_dim: Latent spatial channel dimension (default: 48).
        t_in: Number of historical input timesteps (default: 4).
        num_leads: Number of future forecast lead times (default: 5).
        num_translator_blocks: Depth of translator stack (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
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

        # 1. Spatial Encoder: Encodes each frame (B * T_in, C, H, W) -> (B * T_in, hidden_dim, H, W)
        self.encoder = nn.Sequential(
            BasicConv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            BasicConv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        )

        # 2. Spatiotemporal Translator: Maps (B, T_in * hidden_dim, H, W) -> (B, num_leads * hidden_dim, H, W)
        translator_in_channels = t_in * hidden_dim
        translator_out_channels = num_leads * hidden_dim

        self.trans_in = BasicConv2d(translator_in_channels, translator_out_channels, kernel_size=1, padding=0)
        translator_layers = []
        for _ in range(num_translator_blocks):
            translator_layers.append(GatedTemporalBlock(translator_out_channels))
        self.translator = nn.Sequential(*translator_layers)

        # 3. Spatial Decoder: Multi-head prediction for occurrence logits and continuous QPF
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
        """Forward pass through SimVP spatiotemporal prediction pipeline.

        Args:
            x_seq: (B, T_in, C, H, W) input tensor.

        Returns:
            Tuple of:
              - cat_logits: (B, num_leads, H, W) occurrence logits
              - qpf_rates: (B, num_leads, H, W) continuous rain rates (mm/hr)
        """
        B, T, C, H, W = x_seq.shape

        # Step 1: Spatial Encoding
        # Reshape to (B * T, C, H, W)
        x_flat = x_seq.view(B * T, C, H, W)
        enc_feat = self.encoder(x_flat)  # (B * T, hidden_dim, H, W)

        # Reshape to (B, T * hidden_dim, H, W)
        enc_feat = enc_feat.view(B, T * self.hidden_dim, H, W)

        # Step 2: Spatiotemporal Translation
        trans_feat = self.trans_in(enc_feat)  # (B, num_leads * hidden_dim, H, W)
        trans_feat = self.translator(trans_feat)

        # Reshape to (B, num_leads, hidden_dim, H, W)
        lead_feats = trans_feat.view(B, self.num_leads, self.hidden_dim, H, W)

        # Step 3: Multi-Head Spatial Decoding
        cat_outs = []
        qpf_outs = []
        for k in range(self.num_leads):
            h_k = lead_feats[:, k]
            cat_outs.append(self.cat_heads[k](h_k).squeeze(1))
            qpf_outs.append(self.qpf_heads[k](h_k).squeeze(1))

        return torch.stack(cat_outs, dim=1), torch.stack(qpf_outs, dim=1)
