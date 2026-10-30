"""Baseline Models and Optical Flow Extrapolators for Weather Nowcasting.

Contains verified implementations of:
  - MultiTaskWeatherFCLSTM (Shi et al., 2015 adapted)
  - MultiTaskWeatherMLP (Spatiotemporal Deep MLP)
  - Sparse Lucas-Kanade Optical Flow (rainymotion inspired)
  - Dense Horn-Schunck Variational Optical Flow
  - Semi-Lagrangian Advection Extrapolator
  - 16-D Spatiotemporal Feature Extractor for Logistic Regression
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn


class MultiTaskWeatherFCLSTM(nn.Module):
    """Dimensionally-Projected Fully Connected LSTM (Shi et al., 2015 adapted)."""

    def __init__(
        self,
        in_features: int = 3 * 25 * 40,
        bottleneck: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_leads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.spatial_dim = 25 * 40
        self.proj = nn.Sequential(
            nn.Linear(in_features, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=bottleneck,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        cat_heads = []
        qpf_heads = []
        for _ in range(num_leads):
            cat_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(256, self.spatial_dim),
                )
            )
            qpf_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(256, self.spatial_dim),
                    nn.Softplus(),
                )
            )
        self.cat_heads = nn.ModuleList(cat_heads)
        self.qpf_heads = nn.ModuleList(qpf_heads)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x_seq.shape
        flat_seq = x_seq.view(B, T, C * H * W)
        proj_seq = self.proj(flat_seq)
        _, (h_n, _) = self.lstm(proj_seq)
        last_h = h_n[-1]  # (B, hidden_size)

        cat_out = [head(last_h).view(B, 1, H, W) for head in self.cat_heads]
        qpf_out = [head(last_h).view(B, 1, H, W) for head in self.qpf_heads]
        return torch.cat(cat_out, dim=1), torch.cat(qpf_out, dim=1)


class MultiTaskWeatherMLP(nn.Module):
    """Multi-Head Spatiotemporal Deep MLP."""

    def __init__(self, in_channels: int = 16, num_leads: int = 5, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cat_heads = nn.ModuleList([nn.Conv2d(32, 1, kernel_size=1) for _ in range(num_leads)])
        self.qpf_heads = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Softplus()) for _ in range(num_leads)]
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = feat.shape
        h = self.backbone(feat)
        cat_out = [head(h) for head in self.cat_heads]
        qpf_out = [head(h) for head in self.qpf_heads]
        return torch.cat(cat_out, dim=1), torch.cat(qpf_out, dim=1)


def extract_spatiotemporal_features(x_raw: np.ndarray, tb_default: float = 258.43) -> np.ndarray:
    """Extract 16-D spatiotemporal feature tensor from raw 4-step sequence (Kelvin and mm/hr)."""
    T, C, H, W = x_raw.shape
    tb_raw = x_raw[:, 0]
    p_raw = x_raw[:, 1]
    m_cur = x_raw[-1, 2]

    tb_seq = np.where(np.isfinite(tb_raw) & (tb_raw > 150.0), tb_raw, tb_default).astype(np.float32)
    p_seq = np.where(np.isfinite(p_raw) & (p_raw >= 0.0), p_raw, 0.0).astype(np.float32)
    m_cur = np.nan_to_num(m_cur, nan=0.0).astype(np.float32)

    feats = []
    # 1..4: Past 4 precip
    for t in range(T):
        feats.append(p_seq[t])
    # 5..8: Past 4 IR Tb
    for t in range(T):
        feats.append(tb_seq[t])
    # 9: Validity mask
    feats.append(m_cur)
    # 10: IR cooling rate
    feats.append(tb_seq[-1] - tb_seq[-2])
    # 11: Precip tendency
    feats.append(p_seq[-1] - p_seq[-2])
    # 12..14: 3x3 local precip stats
    p_cur = p_seq[-1]
    feats.append(ndi.uniform_filter(p_cur, size=3))
    feats.append(ndi.maximum_filter(p_cur, size=3))
    p_mean = ndi.uniform_filter(p_cur, size=3)
    p_sq_mean = ndi.uniform_filter(p_cur**2, size=3)
    feats.append(np.sqrt(np.maximum(p_sq_mean - p_mean**2, 0.0)))
    # 15..16: 3x3 local IR stats
    tb_cur = tb_seq[-1]
    feats.append(ndi.uniform_filter(tb_cur, size=3))
    feats.append(ndi.minimum_filter(tb_cur, size=3))

    res = np.stack(feats, axis=0).astype(np.float32)
    return np.nan_to_num(res, nan=0.0)


def compute_sparse_lucas_kanade_flow(i1: np.ndarray, i2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse Lucas-Kanade feature motion tracking."""
    s1 = ndi.gaussian_filter(np.nan_to_num(i1, nan=0.0), sigma=0.8)
    s2 = ndi.gaussian_filter(np.nan_to_num(i2, nan=0.0), sigma=0.8)
    it = s2 - s1
    iy, ix = np.gradient(s2)

    win_size = 5
    a11 = ndi.uniform_filter(ix * ix, size=win_size)
    a12 = ndi.uniform_filter(ix * iy, size=win_size)
    a22 = ndi.uniform_filter(iy * iy, size=win_size)
    b1 = ndi.uniform_filter(-ix * it, size=win_size)
    b2 = ndi.uniform_filter(-iy * it, size=win_size)

    det = a11 * a22 - a12 * a12 + 1e-3
    u = (a22 * b1 - a12 * b2) / det
    v = (-a12 * b1 + a11 * b2) / det

    # Feature mask: precipitative storm peaks >= 1.0 mm/hr
    feat_mask = s2 >= 1.0
    if np.any(feat_mask):
        u_med = float(np.median(u[feat_mask]))
        v_med = float(np.median(v[feat_mask]))
    else:
        u_med, v_med = 0.0, 0.0

    u_field = np.full_like(s2, np.clip(u_med, -5.0, 5.0), dtype=np.float32)
    v_field = np.full_like(s2, np.clip(v_med, -5.0, 5.0), dtype=np.float32)
    return u_field, v_field


def compute_dense_horn_schunck_flow(
    i1: np.ndarray,
    i2: np.ndarray,
    alpha: float = 1.0,
    num_iter: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense Horn-Schunck global variational optical flow with smoothness constraints."""
    s1 = ndi.gaussian_filter(np.nan_to_num(i1, nan=0.0), sigma=1.0)
    s2 = ndi.gaussian_filter(np.nan_to_num(i2, nan=0.0), sigma=1.0)
    it = s2 - s1
    iy, ix = np.gradient(s2)

    kernel = np.array(
        [[1 / 12, 1 / 6, 1 / 12],
         [1 / 6, 0.0, 1 / 6],
         [1 / 12, 1 / 6, 1 / 12]],
        dtype=np.float32,
    )

    u = np.zeros_like(s2, dtype=np.float32)
    v = np.zeros_like(s2, dtype=np.float32)
    denom = alpha**2 + ix**2 + iy**2 + 1e-5

    for _ in range(num_iter):
        u_avg = ndi.convolve(u, kernel, mode="reflect")
        v_avg = ndi.convolve(v, kernel, mode="reflect")
        num = ix * u_avg + iy * v_avg + it
        u = u_avg - ix * (num / denom)
        v = v_avg - iy * (num / denom)

    u = np.clip(u, -5.0, 5.0).astype(np.float32)
    v = np.clip(v, -5.0, 5.0).astype(np.float32)
    return u, v


def semi_lagrangian_advect(img: np.ndarray, u: np.ndarray, v: np.ndarray, steps: int = 1) -> np.ndarray:
    """Backward semi-Lagrangian warping extrapolation."""
    H, W = img.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_warp = xx - steps * u
    y_warp = yy - steps * v
    coords = np.array([y_warp, x_warp])
    warped = ndi.map_coordinates(
        np.nan_to_num(img, nan=0.0),
        coords,
        order=1,
        mode="nearest",
        cval=0.0,
    )
    return np.maximum(warped, 0.0).astype(np.float32)
