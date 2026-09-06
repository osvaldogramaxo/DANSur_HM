"""
Neural network modules and layers for DANSur.
"""

from typing import List, Optional, Type
import torch
from torch import nn


class SinActivation(nn.Module):
    """
    Sinusoidal activation function scaled to [0, 1].
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) / 2.0 + 0.5


class resLin(nn.Module):
    """
    Residual linear block with optional LayerNorm and activation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        use_norm: bool = False
    ):
        super(resLin, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.downproj = nn.Linear(in_features=out_features, out_features=in_features, bias=bias)
        self.activation = activation
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.use_norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        out = self.downproj(out)
        return x + out


class nnPCA(nn.Module):
    """
    Project input onto amplitude and phase PCA basis.
    """
    def __init__(
        self,
        amp_basis: torch.Tensor,
        amp_mean: torch.Tensor,
        phase_basis: torch.Tensor,
        phase_mean: torch.Tensor,
        device: str = 'cuda:0'
    ):
        super(nnPCA, self).__init__()
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_ = (x[:, :len(self.amp_basis.T)].to(self.device) - self.amp_mean.to(self.device)).T
        p_ = (x[:, len(self.amp_basis.T):].to(self.device) - self.phase_mean.to(self.device)).T
        proj_a = torch.matmul(self.amp_basis.to(self.device), a_)
        proj_p = torch.matmul(self.phase_basis.to(self.device), p_)
        out = torch.cat([proj_a, proj_p], dim=0).to(self.device).T
        return out


class invPCA(nn.Module):
    """
    Inverse PCA projection layer reconstructing amplitude and phase from basis coefficients.
    """
    def __init__(
        self,
        amp_basis: torch.Tensor,
        amp_mean: torch.Tensor,
        phase_basis: torch.Tensor,
        phase_mean: torch.Tensor,
        device: str = 'cuda:0'
    ):
        super(invPCA, self).__init__()
        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_ = x[:, :len(self.amp_basis)].to(self.device)
        p_ = x[:, len(self.amp_basis):].to(self.device)
        invproj_a = torch.matmul(a_, self.amp_basis.to(self.device)) + self.amp_mean.to(self.device)
        invproj_p = torch.matmul(p_, self.phase_basis.to(self.device)) + self.phase_mean.to(self.device)
        out = torch.cat([invproj_a, invproj_p], dim=1).to(self.device)
        return out


class Decoder(nn.Module):
    """
    Neural network decoder predicting amplitude and phase basis coefficients from latent/physical parameters.
    """
    def __init__(
        self,
        latent_dim: int,
        amp_basis: torch.Tensor,
        amp_mean: torch.Tensor,
        phase_basis: torch.Tensor,
        phase_mean: torch.Tensor,
        layers: List[int] = [128, 256, 1024, 1024],
        act_fn: Type[nn.Module] = nn.ReLU,
        block: Type[nn.Module] = nn.Linear,
        device=None
    ):
        super(Decoder, self).__init__()
        self.device = device

        self.base_dim = amp_basis.shape[0] + phase_basis.shape[0]
        self.latent_dim = latent_dim
        self.amp_dim = amp_basis.shape[0]
        self.phase_dim = phase_basis.shape[0]
        self.block = block
        self.act_fn = act_fn

        self.amp_basis = nn.Parameter(amp_basis.to(device), requires_grad=False)
        self.amp_mean = nn.Parameter(amp_mean.to(device), requires_grad=False)
        self.phase_basis = nn.Parameter(phase_basis.to(device), requires_grad=False)
        self.phase_mean = nn.Parameter(phase_mean.to(device), requires_grad=False)

        self.decoder = self._build_decoder(layers).to(self.device)
        self.to(self.device)

    def _build_decoder(self, layers: List[int]) -> nn.Sequential:
        layers_list = []
        for i, current_layer in enumerate(layers):
            if i == 0:
                layers_list.append(self.block(self.latent_dim, current_layer).to(self.device))
            else:
                layers_list.extend([self.act_fn(), self.block(layers[i-1], current_layer).to(self.device)])
        layers_list.extend([self.act_fn(), self.block(layers[-1], self.base_dim).to(self.device)])
        return nn.Sequential(*layers_list)

    def PCA(self, x: torch.Tensor) -> torch.Tensor:
        a_ = (x[..., :len(self.amp_basis.T)].to(self.device) - self.amp_mean.to(self.device)).T
        p_ = (x[..., len(self.amp_basis.T):].to(self.device) - self.phase_mean.to(self.device)).T
        proj_a = torch.matmul(self.amp_basis.to(self.device), a_)
        proj_p = torch.matmul(self.phase_basis.to(self.device), p_)
        out = torch.cat([proj_a, proj_p], dim=0).to(self.device).T
        return out

    def invPCA(self, x: torch.Tensor) -> torch.Tensor:
        a_ = x[..., :len(self.amp_basis)].to(self.device)
        p_ = x[..., len(self.amp_basis):].to(self.device)
        invproj_a = torch.matmul(a_, self.amp_basis.to(self.device)) + self.amp_mean.to(self.device)
        invproj_p = torch.matmul(p_, self.phase_basis.to(self.device)) + self.phase_mean.to(self.device)
        out = torch.cat([invproj_a, invproj_p], dim=1).to(self.device)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = self.invPCA(x)
        return x
