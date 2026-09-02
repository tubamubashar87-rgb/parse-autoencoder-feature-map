"""
Sparse Autoencoder for dictionary learning over transformer residual stream
activations.

Formulation
-----------
Given an activation vector x in R^d_model, the SAE learns an overcomplete
dictionary of d_hidden >> d_model unit-norm feature directions and encodes x
as a sparse non-negative combination of those directions:

    f  = ReLU(W_enc (x - b_dec) + b_enc)          feature activations
    x_hat = W_dec f + b_dec                        reconstruction

    L_SAE = || x - x_hat ||_2^2  +  lambda * || f ||_1

The reconstruction term is a per-sample MSE-style L2 term and the L1 term is
the standard sparsity-inducing penalty from dictionary learning (Olshausen &
Field, 1997) as applied to transformer internals in Cunningham et al. (2023)
and Bricken et al. (2023). W_dec columns are re-normalized to unit norm so
that the L1 penalty cannot be trivially deflated by shrinking the decoder
weights while inflating the encoder (the standard SAE "norm gaming" failure
mode).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Single hidden layer, tied-bias sparse autoencoder.

    Args:
        d_model: dimensionality of the input activation vector.
        d_hidden: size of the overcomplete feature dictionary. Typically
            4x-32x d_model for interpretability work.
        l1_coefficient: lambda, the weight on the sparsity penalty.
        tied_weights: if True, initialize W_dec = W_enc^T (standard practice,
            weights are allowed to untie during training since they are
            separate parameters).
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        l1_coefficient: float = 1e-3,
        tied_weights: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l1_coefficient = l1_coefficient

        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        w_dec = torch.randn(d_hidden, d_model)
        w_dec = w_dec / w_dec.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(w_dec)

        if tied_weights:
            self.W_enc = nn.Parameter(w_dec.t().clone())
        else:
            w_enc = torch.randn(d_model, d_hidden) * (1.0 / d_model**0.5)
            self.W_enc = nn.Parameter(w_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> f: (..., d_hidden), f >= 0."""
        x_centered = x - self.b_dec
        pre_activation = x_centered @ self.W_enc + self.b_enc
        return torch.relu(pre_activation)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """f: (..., d_hidden) -> x_hat: (..., d_model)."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass, returns reconstruction, features, and losses.

        x is expected to be shape (batch, d_model) or (batch, seq, d_model);
        all loss terms are computed per-token and averaged over every
        leading dimension, so both shapes work without modification.
        """
        f = self.encode(x)
        x_hat = self.decode(f)

        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        l1_loss = f.abs().sum(dim=-1).mean()
        loss = recon_loss + self.l1_coefficient * l1_loss

        with torch.no_grad():
            l0 = (f > 0).float().sum(dim=-1).mean()
            variance = x.var(dim=0).sum().clamp_min(1e-8)
            residual_variance = (x - x_hat).var(dim=0).sum()
            fvu = residual_variance / variance  # fraction of variance unexplained

        return {
            "x_hat": x_hat,
            "features": f,
            "loss": loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "l0": l0,
            "fvu": fvu,
        }

    @torch.no_grad()
    def normalize_decoder_weights(self) -> None:
        """Renormalize each decoder feature direction to unit L2 norm.

        Called after every optimizer step. Without this, gradient descent can
        reduce the L1 penalty by shrinking W_dec norms and inflating W_enc /
        f proportionally, which changes nothing about the reconstruction but
        makes the sparsity penalty meaningless.
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_gradient_component(self) -> None:
        """Project out the component of W_dec.grad parallel to W_dec itself.

        Standard trick (Bricken et al. 2023 / Anthropic SAE post): if we only
        renormalize after the step, the optimizer's momentum still tries to
        grow the norm indefinitely, wasting gradient signal. Removing the
        radial component of the gradient before the step keeps updates
        confined to the unit sphere.
        """
        if self.W_dec.grad is None:
            return
        w = self.W_dec
        g = self.W_dec.grad
        parallel_component = (g * w).sum(dim=-1, keepdim=True)
        g -= parallel_component * w / w.norm(dim=-1, keepdim=True).pow(2).clamp_min(1e-8)

    @torch.no_grad()
    def feature_density(self, f: torch.Tensor) -> torch.Tensor:
        """Fraction of a batch of tokens on which each feature fires.

        f: (n_tokens, d_hidden) -> (d_hidden,) in [0, 1].
        """
        return (f > 0).float().mean(dim=0)
