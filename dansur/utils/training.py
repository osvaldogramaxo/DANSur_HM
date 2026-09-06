"""
Training loops, EarlyStopping callback, and polynomial basis/filter functions.
"""

from typing import Any, Callable, List, Optional
import numpy as np
from scipy.special import eval_legendre
from sympy import Symbol, chebyshevt, lambdify
import torch


# ========== Training Callbacks and Loops ==========

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves model checkpoint when improved.
    """
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = 'checkpoint.pt',
        trace_func: Callable = print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
    verbose: int = 0,
    loss_fn: Optional[Callable] = None,
    lr_schedule: Optional[Any] = None,
    post_proc: Callable = lambda args: args
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = post_proc(output)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if lr_schedule is not None:
            lr_schedule.step()

        if verbose > 0 and batch_idx % verbose == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return train_loss / len(train_loader)


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    verbose: int = 0,
    loss_fn: Optional[Callable] = None,
    post_proc: Callable = lambda args: args
) -> float:
    """
    Evaluate model on validation/test DataLoader.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = post_proc(output)
            test_loss += loss_fn(output, target).item()

    test_loss /= len(test_loader)
    if verbose > 0:
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return test_loss


# ========== Polynomial Basis and Filtering Functions ==========

def legendreDer(k: int, x: np.ndarray) -> np.ndarray:
    """
    Compute derivatives of Legendre polynomials.
    """
    def _legendre(k_order, val):
        return (2 * k_order + 1) * eval_legendre(k_order, val)

    out = np.zeros((k, x.shape[0]))
    for ki in range(k):
        for i in range(x.shape[0]):
            out[ki, i] = _legendre(ki + 1, x[i])
    return out.T


def phi_(phi_c: float, x: np.ndarray, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    """
    Phase transformation function.
    """
    return 0.5 * (1.0 - phi_c) * (1.0 - np.cos(np.pi * (x - lb) / (ub - lb))) + phi_c * (x - lb) / (ub - lb)


def get_phi_psi(k: int, base: str) -> List[Any]:
    """
    Get basis functions for specified polynomial order and base ('legendre', 'chebyshev', 'sin').
    """
    x = Symbol('x')
    phi = [0] * k
    phi[0] = 1.0 + 0.0 * x
    if k > 0:
        if base == 'legendre':
            phi[1] = x
            for i in range(1, k):
                phi[i + 1] = (2 * i + 1) / (i + 1) * x * phi[i] - i / (i + 1) * phi[i - 1]
            phi = phi[1:]
        elif base == 'chebyshev':
            for n in range(k):
                phi[n] = chebyshevt(n, x)
            phi = phi[1:]
        elif base == 'sin':
            for i in range(k):
                phi[i] = np.sin(2 * np.pi * (i + 1) * x) / (2 * (i + 1) * np.pi**2)
        else:
            raise ValueError(f"Invalid basis: '{base}', expected 'legendre', 'chebyshev', or 'sin'")
    return phi


def get_filter(base: str, k: int) -> np.ndarray:
    """
    Compute basis transformation/filtering matrix for spectral representation.
    """
    def psi(n, x_sym):
        w = 1.0
        for i in range(k):
            w *= (x_sym - i) / (i + 1)
        return w * x_sym ** (n - k)

    x = Symbol('x')
    psi_list = [0] * k
    for i in range(k):
        psi_list[i] = psi(i + 1, x)

    if base not in ['legendre', 'chebyshev']:
        raise ValueError(f"Invalid basis: '{base}', expected 'legendre' or 'chebyshev'")

    mat_phi = get_phi_psi(k, base)
    mat_psi = np.zeros((k, k), dtype=float)

    for i in range(k):
        for j in range(k):
            integrand = mat_phi[i] * psi_list[j]
            integrand = lambdify(x, integrand, 'numpy')
            mat_psi[i, j] = integrand(1.0) - integrand(0.0)

    return mat_psi
