# -*- coding: utf-8 -*-
"""
Channel models with optional transmitter-side fading context.

This file is a drop-in replacement for channel.py and adds frequency-selective
(pixel-wise) fading support. Each spatial location experiences independent fading,
allowing the Channel-Aware Full FIS model to adapt power allocation to local
channel conditions.

Key idea
--------
- AWGN: context is derived from the nominal SNR only.
- Rayleigh / Rician (Frequency-Selective): context is derived from the instantaneous
  fading power |h|^2 at each spatial location (i,j). The fading map has shape
  [B, 1, H, W] to match the latent tensor dimensions.
- Rician: same as Rayleigh but with a deterministic LOS component.
- The same sampled fading coefficients are reused in the subsequent forward pass,
  so the controller sees the same fading realization that is applied by the channel.

Supported channels
------------------
- AWGN
- Rayleigh (frequency-selective / pixel-wise)
- Rician (frequency-selective / pixel-wise with LOS)
- rayleigh_legacy (retained only for reproducibility, block fading)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(
        self,
        channel_type: str = "awgn",
        P: float = 1.0,
        snr_db: float = 13.0,
        eps: float = 1e-8,
        rician_k: float = 4.0,
        context_db_min: float = -60.0,
        context_db_max: float = 25.0,
    ):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        self.P = float(P)
        self.snr_db = float(snr_db)
        self.eps = float(eps)
        self.rician_k = float(rician_k)
        self.context_db_min = float(context_db_min)
        self.context_db_max = float(context_db_max)

        self._fading_equalize = False
        self._pending_fading: Optional[Tuple[str, torch.Tensor, torch.Tensor]] = None
        self.last_context: Dict[str, torch.Tensor] = {}

    def change_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    def change_rician_k(self, rician_k: float) -> None:
        self.rician_k = float(rician_k)

    def enable_rayleigh_equalization(self, enable: bool = True) -> None:
        self._fading_equalize = bool(enable)

    def enable_fading_equalization(self, enable: bool = True) -> None:
        self._fading_equalize = bool(enable)

    def _sigma(self, device, dtype=torch.float32) -> torch.Tensor:
        snr = 10.0 ** (self.snr_db / 10.0)
        sigma = math.sqrt(self.P / snr / 2.0)
        return torch.tensor(sigma, device=device, dtype=dtype)

    def _snr_lin(self, device, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(10.0 ** (self.snr_db / 10.0), device=device, dtype=dtype)

    def _norm_db(self, x_db: torch.Tensor) -> torch.Tensor:
        x = (x_db - self.context_db_min) / (self.context_db_max - self.context_db_min + self.eps)
        return torch.clamp(x, 0.0, 1.0)

    def _split_iq(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError("Channel expects even channel dimension (I/Q halves).")
        C2 = C // 2
        return x[:, :C2], x[:, C2:]

    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(x.device, x.dtype)
        noise = sigma * torch.randn_like(x)
        return x + noise

    def _apply_complex_fading(self, x: torch.Tensor, hI: torch.Tensor, hQ: torch.Tensor) -> torch.Tensor:
        """
        Apply complex fading with spatial maps hI, hQ of shape [B, 1, H, W].
        """
        xI, xQ = self._split_iq(x)
        yI = xI * hI - xQ * hQ
        yQ = xI * hQ + xQ * hI
        sigma = self._sigma(x.device, x.dtype)
        yI = yI + sigma * torch.randn_like(yI)
        yQ = yQ + sigma * torch.randn_like(yQ)

        if self._fading_equalize:
            denom = (hI * hI + hQ * hQ).clamp_min(self.eps)
            xI_hat = (yI * hI + yQ * hQ) / denom
            xQ_hat = (yQ * hI - yI * hQ) / denom
            return torch.cat([xI_hat, xQ_hat], dim=1)
        return torch.cat([yI, yQ], dim=1)

    def _sample_rayleigh(self, batch_size: int, H: int, W: int, device, dtype):
        """
        Sample frequency-selective Rayleigh fading coefficients.
        Returns hI, hQ of shape [B, 1, H, W] - one fading coefficient per spatial location.
        """
        hI = torch.randn(batch_size, 1, H, W, device=device, dtype=dtype) / math.sqrt(2.0)
        hQ = torch.randn(batch_size, 1, H, W, device=device, dtype=dtype) / math.sqrt(2.0)
        return hI, hQ

    def _sample_rician(self, batch_size: int, H: int, W: int, device, dtype):
        """
        Sample frequency-selective Rician fading coefficients with LOS component.
        Returns hI, hQ of shape [B, 1, H, W].
        """
        K = max(float(self.rician_k), 0.0)
        theta = 2.0 * math.pi * torch.rand(batch_size, 1, H, W, device=device, dtype=dtype)
        h_los_I = torch.cos(theta)
        h_los_Q = torch.sin(theta)
        h_nlos_I = torch.randn(batch_size, 1, H, W, device=device, dtype=dtype) / math.sqrt(2.0)
        h_nlos_Q = torch.randn(batch_size, 1, H, W, device=device, dtype=dtype) / math.sqrt(2.0)
        los_scale = math.sqrt(K / (K + 1.0)) if K > 0.0 else 0.0
        nlos_scale = math.sqrt(1.0 / (K + 1.0))
        hI = los_scale * h_los_I + nlos_scale * h_nlos_I
        hQ = los_scale * h_los_Q + nlos_scale * h_nlos_Q
        return hI, hQ

    def _sample_legacy_rayleigh(self, batch_size: int, device, dtype):
        """Legacy block-fading: single scalar per batch [B, 1, 1, 1]."""
        h0 = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        h1 = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        return h0, h1

    @torch.no_grad()
    def sample_context(self, batch_size: int, H: int, W: int, device, dtype=torch.float32) -> Dict[str, torch.Tensor]:
        """
        Sample a per-spatial-location channel reliability context for the transmitter.

        Returns a dictionary containing spatial tensors of shape [B, 1, H, W]:
        - gamma_eff_lin: instantaneous effective SNR in linear scale (per location)
        - gamma_eff_db: same quantity in dB (per location)
        - gamma_eff_norm: normalized to [0,1] for the FIS antecedent (per location)
        - h_abs2: instantaneous channel power (AWGN -> 1)
        - channel_rel: normalized effective SNR for controller [B, 1, H, W]

        For fading channels, the sampled coefficients are cached and reused in the
        next forward() call.
        """
        ct = self.channel_type
        sigma = self._sigma(device, dtype)
        noise_var = 2.0 * sigma * sigma
        snr_lin = self._snr_lin(device, dtype)

        if ct in ("awgn",):
            h_abs2 = torch.ones(batch_size, 1, H, W, device=device, dtype=dtype)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1).expand(batch_size, 1, H, W)
            self._pending_fading = None
        elif ct in ("rayleigh",):
            hI, hQ = self._sample_rayleigh(batch_size, H, W, device, dtype)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)
        elif ct in ("rician",):
            hI, hQ = self._sample_rician(batch_size, H, W, device, dtype)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)
        elif ct in ("rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy"):
            h0, h1 = self._sample_legacy_rayleigh(batch_size, device, dtype)
            h_abs2_scalar = 0.5 * (h0 * h0 + h1 * h1).clamp_min(self.eps)
            h_abs2 = h_abs2_scalar.expand(batch_size, 1, H, W)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("legacy", h0, h1)
        else:
            raise ValueError(
                f"Unsupported channel_type='{self.channel_type}'. "
                "Use 'awgn', 'rayleigh', 'rician', or 'rayleigh_legacy'."
            )

        gamma_eff_db = 10.0 * torch.log10(gamma_eff_lin.clamp_min(self.eps))
        gamma_eff_norm = self._norm_db(gamma_eff_db)
        posteq_noise_var = noise_var / h_abs2
        eq_quality = 1.0 / posteq_noise_var.clamp_min(self.eps)

        ctx = {
            "gamma_eff_lin": gamma_eff_lin,
            "gamma_eff_db": gamma_eff_db,
            "gamma_eff_norm": gamma_eff_norm,
            "h_abs2": h_abs2,
            "posteq_noise_var": posteq_noise_var,
            "eq_quality": eq_quality,
            "channel_rel": gamma_eff_norm,
            "channel_type": ct,
            "fading_equalize": torch.tensor(float(self._fading_equalize), device=device, dtype=dtype),
            "spatial_shape": (batch_size, 1, H, W),
        }
        self.last_context = ctx
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = self.channel_type
        B, C, H, W = x.shape

        if ct in ("awgn",):
            return self._awgn(x)

        if ct in ("rayleigh", "rician"):
            if self._pending_fading is not None:
                kind, hI, hQ = self._pending_fading
                self._pending_fading = None
                if kind != "complex":
                    raise RuntimeError("Pending fading kind mismatch for complex channel.")
                if hI.shape[2:] != (H, W):
                    hI = torch.nn.functional.interpolate(
                        hI, size=(H, W), mode='bilinear', align_corners=False
                    )
                    hQ = torch.nn.functional.interpolate(
                        hQ, size=(H, W), mode='bilinear', align_corners=False
                    )
            else:
                if ct == "rayleigh":
                    hI, hQ = self._sample_rayleigh(B, H, W, x.device, x.dtype)
                else:
                    hI, hQ = self._sample_rician(B, H, W, x.device, x.dtype)
            return self._apply_complex_fading(x, hI, hQ)

        if ct in ("rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy"):
            if self._pending_fading is not None:
                kind, h0, h1 = self._pending_fading
                self._pending_fading = None
                if kind != "legacy":
                    raise RuntimeError("Pending fading kind mismatch for legacy channel.")
            else:
                h0, h1 = self._sample_legacy_rayleigh(B, x.device, x.dtype)

            C2 = C // 2
            xI, xQ = x[:, :C2], x[:, C2:]
            yI = xI * h0
            yQ = xQ * h1
            sigma = self._sigma(x.device, x.dtype)
            yI = yI + sigma * torch.randn_like(yI)
            yQ = yQ + sigma * torch.randn_like(yQ)
            if self._fading_equalize:
                yI = yI / h0.clamp_min(self.eps)
                yQ = yQ / h1.clamp_min(self.eps)
            return torch.cat([yI, yQ], dim=1)

        raise ValueError(
            f"Unsupported channel_type='{self.channel_type}'. "
            "Use 'awgn', 'rayleigh', 'rician', or 'rayleigh_legacy'."
        )
