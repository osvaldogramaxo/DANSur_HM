"""
Loss functions for DANSur neural network and waveform optimization.
"""

from typing import Dict, Tuple
import torch
from torch import nn

from .waveforms import Cover_Sphere, mymismatch


def weighted_mse_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean squared error loss.
    """
    return torch.mean(weight * (input - target) ** 2)


def weighted_L1_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Weighted L1 loss.
    """
    return torch.mean(weight * torch.abs(input - target))


class ASDL1Loss(nn.Module):
    """
    L1 loss between predicted and ground-truth waveforms in Amplitude Spectral Density (ASD) domain.
    """
    def __init__(self, reduction: str = 'mean', scale: str = 'linear', normalize: bool = False):
        super(ASDL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.scale = scale
        self.normalize = normalize

    def forward(self, vwf_wave: torch.Tensor, outputs_wave_valid: torch.Tensor) -> torch.Tensor:
        gt_fft = torch.fft.fft(vwf_wave)
        pred_fft = torch.fft.fft(outputs_wave_valid)
        if self.normalize:
            gt_fft = gt_fft / torch.mean(torch.abs(gt_fft))
            pred_fft = pred_fft / torch.mean(torch.abs(pred_fft))

        if self.scale == 'dB':
            gt_asd = 20.0 * torch.log10(torch.abs(gt_fft))
            pred_asd = 20.0 * torch.log10(torch.abs(pred_fft))
        elif self.scale == 'linear':
            gt_asd = torch.abs(gt_fft)
            pred_asd = torch.abs(pred_fft)
        else:
            raise ValueError(f"Unknown scale: {self.scale}, expected 'linear' or 'dB'")

        return self.l1_loss(gt_asd, pred_asd)


class HMLoss(nn.Module):
    """
    Higher-order mode loss combining spherical-harmonic mismatch and power difference.
    """
    def __init__(self, modes: Dict[int, Tuple[int, int]], device: str = 'cpu'):
        super(HMLoss, self).__init__()
        self.modes = modes
        self.device = device
        self.cover_forward = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)
        self.cover_forward_sxs = Cover_Sphere(modes, n=5, infer_neg_m=True, device=device)

    def forward(self, pred: torch.Tensor, wf: torch.Tensor) -> torch.Tensor:
        wf_wave = self.cover_forward_sxs(wf).to(self.device).flatten(0, 2)
        outputs_wave = self.cover_forward(pred).flatten(0, 2)
        mm_loss = mymismatch(outputs_wave, wf_wave)
        mm_loss = torch.nan_to_num(mm_loss)
        wave_power = 1.0
        power_diff = nn.L1Loss()(torch.abs(wf_wave).sum(dim=-1), torch.abs(outputs_wave).sum(dim=-1))
        loss = torch.log10((mm_loss * wave_power).mean()) + power_diff.mean()
        return loss


class SXSLoss(HMLoss):
    """
    Loss function for training/fine-tuning on SXS waveform data (inherits from HMLoss).
    """
    pass


class LpLoss(object):
    """
    Relative or absolute Lp loss between two tensors.
    """
    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True):
        assert d > 0 and p > 0, "Dimension and Lp-norm type must be positive"
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        x = x.view(num_examples, -1)
        y = y.view(num_examples, -1)
        all_norms = (h ** (self.d / self.p)) * torch.norm(x - y, self.p, 1)
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)
