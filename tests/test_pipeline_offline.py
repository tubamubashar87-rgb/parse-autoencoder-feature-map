"""
Offline sanity test that exercises the full activation-collection + SAE
training loop against a tiny hand-built transformer stand-in, so it runs
without any network access or GPT-2 weights. This is a debugging harness,
not the project's public test surface.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.activation_store import ActivationStore, collect_activations
from src.model import SparseAutoencoder


class DummyBlock(nn.Module):
    """Mimics a GPT2Block: returns a tuple (hidden_states, ...) like HF does."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.tanh(self.proj(x))
        return (out, None)  # HF-style tuple output


class DummyTransformer(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.embed = nn.Embedding(1000, d_model)
        self.blocks = nn.ModuleList([DummyBlock(d_model) for _ in range(n_layers)])

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)[0]
        return x


def test_activation_store_shapes():
    d_model = 32
    model = DummyTransformer(d_model=d_model, n_layers=4)
    hook_module = model.blocks[2]

    store = ActivationStore(d_model=d_model, max_tokens=10_000)
    store.register_on(hook_module)

    batch = torch.randint(0, 1000, (5, 7))  # (batch, seq)
    with torch.no_grad():
        model(batch)
    store.remove()

    dataset = store.build_dataset()
    assert dataset.shape == (5 * 7, d_model), dataset.shape
    print("test_activation_store_shapes passed:", dataset.shape)


def test_activation_store_shape_mismatch_raises():
    d_model = 32
    wrong_d_model = 16
    model = DummyTransformer(d_model=d_model, n_layers=2)
    store = ActivationStore(d_model=wrong_d_model, max_tokens=10_000)
    store.register_on(model.blocks[0])

    batch = torch.randint(0, 1000, (2, 4))
    raised = False
    try:
        with torch.no_grad():
            model(batch)
    except ValueError:
        raised = True
    finally:
        store.remove()
    assert raised, "expected shape mismatch to raise ValueError"
    print("test_activation_store_shape_mismatch_raises passed")


def test_collect_activations_helper():
    d_model = 24
    model = DummyTransformer(d_model=d_model, n_layers=3)
    hook_module = model.blocks[1]

    batches = [torch.randint(0, 1000, (4, 6)) for _ in range(5)]
    dataset = collect_activations(
        model=model,
        hook_module=hook_module,
        dataloader=batches,
        d_model=d_model,
        max_tokens=10_000,
        device="cpu",
        forward_fn=lambda m, b: m(b),
    )
    assert dataset.shape == (4 * 6 * 5, d_model), dataset.shape
    print("test_collect_activations_helper passed:", dataset.shape)


def test_collect_activations_respects_max_tokens():
    d_model = 16
    model = DummyTransformer(d_model=d_model, n_layers=2)
    hook_module = model.blocks[0]

    batches = [torch.randint(0, 1000, (10, 10)) for _ in range(20)]  # 100 tokens/batch
    dataset = collect_activations(
        model=model,
        hook_module=hook_module,
        dataloader=batches,
        d_model=d_model,
        max_tokens=250,
        device="cpu",
        forward_fn=lambda m, b: m(b),
    )
    assert dataset.shape[0] == 250, dataset.shape
    print("test_collect_activations_respects_max_tokens passed:", dataset.shape)


def test_sae_forward_shapes_and_loss():
    d_model, d_hidden, n = 16, 64, 100
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coefficient=1e-3)
    x = torch.randn(n, d_model)
    out = sae(x)

    assert out["x_hat"].shape == (n, d_model)
    assert out["features"].shape == (n, d_hidden)
    assert (out["features"] >= 0).all(), "ReLU features must be non-negative"
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    print("test_sae_forward_shapes_and_loss passed. loss =", out["loss"].item())


def test_sae_forward_3d_input():
    """Verify (batch, seq, d_model) inputs work without reshaping externally."""
    d_model, d_hidden = 16, 64
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(3, 5, d_model)
    out = sae(x)
    assert out["x_hat"].shape == (3, 5, d_model)
    assert out["features"].shape == (3, 5, d_hidden)
    print("test_sae_forward_3d_input passed")


def test_decoder_normalization_keeps_unit_norm():
    d_model, d_hidden = 16, 64
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
    # perturb decoder weights to break unit norm, then renormalize
    with torch.no_grad():
        sae.W_dec.mul_(3.7)
    sae.normalize_decoder_weights()
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms
    print("test_decoder_normalization_keeps_unit_norm passed")


def test_gradient_projection_removes_radial_component():
    d_model, d_hidden = 8, 32
    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
    x = torch.randn(50, d_model)
    out = sae(x)
    out["loss"].backward()

    sae.remove_parallel_gradient_component()
    # after projection, grad should have (near) zero component along W_dec direction
    radial = (sae.W_dec.grad * sae.W_dec).sum(dim=-1)
    assert radial.abs().max().item() < 1e-4, radial.abs().max().item()
    print("test_gradient_projection_removes_radial_component passed")


def test_end_to_end_training_reduces_loss():
    """Full pipeline: dummy model -> activation store -> SAE training loop."""
    torch.manual_seed(0)
    d_model = 32
    model = DummyTransformer(d_model=d_model, n_layers=4)
    hook_module = model.blocks[2]

    batches = [torch.randint(0, 1000, (16, 12)) for _ in range(20)]
    activations = collect_activations(
        model=model,
        hook_module=hook_module,
        dataloader=batches,
        d_model=d_model,
        max_tokens=100_000,
        device="cpu",
        forward_fn=lambda m, b: m(b),
    )

    sae = SparseAutoencoder(d_model=d_model, d_hidden=256, l1_coefficient=1e-4)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    losses = []
    for epoch in range(15):
        optimizer.zero_grad()
        out = sae(activations)
        out["loss"].backward()
        sae.remove_parallel_gradient_component()
        optimizer.step()
        sae.normalize_decoder_weights()
        losses.append(out["loss"].item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    print(f"test_end_to_end_training_reduces_loss passed: {losses[0]:.4f} -> {losses[-1]:.4f}")

    with torch.no_grad():
        final = sae(activations)
        density = sae.feature_density(final["features"])
    assert density.shape == (256,)
    print("feature density stats: mean =", density.mean().item(), "dead =", (density == 0).sum().item())


if __name__ == "__main__":
    test_activation_store_shapes()
    test_activation_store_shape_mismatch_raises()
    test_collect_activations_helper()
    test_collect_activations_respects_max_tokens()
    test_sae_forward_shapes_and_loss()
    test_sae_forward_3d_input()
    test_decoder_normalization_keeps_unit_norm()
    test_gradient_projection_removes_radial_component()
    test_end_to_end_training_reduces_loss()
    print("\nALL TESTS PASSED")
