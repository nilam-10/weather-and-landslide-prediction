"""Canonical 2D Convolutional LSTM Implementation for Weather Nowcasting.

Single canonical architecture definition shared across:
  - Model training: scripts/train_weather_convlstm.py & scripts/run_controlled_dataset_size_experiment.py
  - Evaluation: scripts/evaluate_frozen_models_on_independent_test.py
  - FastAPI serving: src/api/main.py
  - Automated testing: tests/test_weather_convlstm.py

Eliminates all duplicated recurrent cell definitions and enforces the exact
Shi et al. (NeurIPS 2015) gate unpacking order: (i, f, g, o).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM Cell (Shi et al., NeurIPS 2015).

    CRITICAL ARCHITECTURE INVARIANT:
    Gates must unpack in the exact canonical order: (i, f, g, o).
      i: Input Gate (sigmoid)
      f: Forget Gate (sigmoid)
      g: Cell Candidate Update (tanh)
      o: Output Gate (sigmoid)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape
        if state is None:
            h = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype,
            )
            c = torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Canonical Shi et al. (2015) gate slice order
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class MultiTaskWeatherConvLSTM(nn.Module):
    """Canonical Multi-Task Weather ConvLSTM with Dual Prediction Heads.

    Heads:
      1. cat_heads: Binary convective precipitation occurrence logits (BCE / Dice)
      2. qpf_heads: Continuous physical rain rate in mm/hr (Smooth L1 + Softplus)

    Parameters:
      in_channels: 3 (IR Tb normalized, log precipitation normalized, availability mask)
      hidden_channels: (32, 32)
      num_leads: 5 (+30m, +60m, +120m, +180m, +360m)
      Total Trainable Parameters: 161,034
      Total State Dict Elements: 161,364
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Union[Tuple[int, ...], List[int]] = (32, 32),
        kernel_size: int = 3,
        num_leads: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.num_leads = num_leads

        cells = []
        c_in = in_channels
        for c_out in hidden_channels:
            cells.append(ConvLSTMCell(c_in, c_out, kernel_size))
            c_in = c_out
        self.cells = nn.ModuleList(cells)
        last_c = hidden_channels[-1]

        # 1. Occurrence Logit Heads (Categorical Classification)
        cat_heads = []
        for _ in range(num_leads):
            cat_heads.append(
                nn.Sequential(
                    nn.Conv2d(last_c, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(16, 1, kernel_size=1),
                )
            )
        self.cat_heads = nn.ModuleList(cat_heads)

        # 2. Continuous Physical QPF Heads (Rain Rate in mm/hr)
        qpf_heads = []
        for _ in range(num_leads):
            qpf_heads.append(
                nn.Sequential(
                    nn.Conv2d(last_c, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(16, 1, kernel_size=1),
                    nn.Softplus(),
                )
            )
        self.qpf_heads = nn.ModuleList(qpf_heads)

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through spatiotemporal recurrent sequence.

        Args:
            x_seq: (B, T, C, H, W) input tensor.

        Returns:
            Tuple of:
              - cat_logits: (B, num_leads, H, W) occurrence logits
              - qpf_rates: (B, num_leads, H, W) continuous rain rates in mm/hr
        """
        B, T, C, H, W = x_seq.shape
        states = [None] * self.num_layers

        for t in range(T):
            cur = x_seq[:, t]
            for l in range(self.num_layers):
                h, c = self.cells[l](cur, states[l])
                states[l] = (h, c)
                cur = h

        last_h = states[-1][0]
        cat_outs = [h(last_h).squeeze(1) for h in self.cat_heads]
        qpf_outs = [h(last_h).squeeze(1) for h in self.qpf_heads]

        return torch.stack(cat_outs, dim=1), torch.stack(qpf_outs, dim=1)


class RuntimeWeatherConvLSTM(nn.Module):
    """Runtime Inference Wrapper for single-task multi-lead ConvLSTM checkpoints.

    Uses canonical ConvLSTMCell with exact (i, f, g, o) Shi et al. (2015) recurrence.
    Matches state-dict keys for:
      - best_convlstm_model.pt (2 heads: +30m, +60m)
      - extended_convlstm_model.pt (4 heads: +30m, +60m, +120m, +180m)
      - multiyear_convlstm_model.pt (5 heads: +30m, +60m, +120m, +180m, +360m)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        num_leads: int = 2,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.cell1 = ConvLSTMCell(in_channels, hidden_channels[0], kernel_size)
        self.cell2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], kernel_size)

        heads = []
        for _ in range(num_leads):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels[1], 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.0),
                    nn.Conv2d(16, 1, kernel_size=1),
                )
            )
        self.lead_heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _, h, w = x.shape
        state1, state2 = None, None
        for t in range(seq_len):
            x_t = x[:, t]
            h1, c1 = self.cell1(x_t, state1)
            state1 = (h1, c1)
            h2, c2 = self.cell2(h1, state2)
            state2 = (h2, c2)
        last_hidden = state2[0]
        outputs = [head(last_hidden).squeeze(1) for head in self.lead_heads]
        return torch.stack(outputs, dim=1)
