"""
Hook-based activation capture for transformer residual stream activations.

The store attaches a forward hook to a single transformer block and records
the residual stream at that point for every token in every batch pushed
through the model. Captured activations are flattened across batch and
sequence dimensions into a flat (n_tokens, d_model) tensor, which is the
input format the SAE (src/model.py) expects.

Supports two backends:
  * HFResidualHook  - hooks a block of a HuggingFace GPT-2 model and reads
                       the residual stream at the output of that block.
  * a generic `register_on` classmethod that works on any nn.Module, so the
    same store can be unit-tested against a toy model without downloading
    GPT-2 weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ActivationStore:
    """Captures activations from one hook point across many forward passes.

    Args:
        d_model: expected last-dimension size of captured activations, used
            only to validate shapes as they arrive.
        max_tokens: stop accumulating once this many tokens have been
            captured (bounds memory use).
        device: device the concatenated dataset tensor is moved to when
            `build_dataset` is called. Individual captures are detached and
            moved to CPU immediately to avoid holding onto the GPU
            autograd graph or GPU memory for the whole collection pass.
    """

    d_model: int
    max_tokens: int = 500_000
    device: str = "cpu"

    _buffer: list[torch.Tensor] = field(default_factory=list, init=False)
    _n_tokens: int = field(default=0, init=False)
    _handle: torch.utils.hooks.RemovableHandle | None = field(default=None, init=False)

    def _hook_fn(self, module: nn.Module, inputs, output):
        # HF GPT2Block returns a tuple whose first element is the residual
        # stream hidden state; plain nn.Modules typically return a tensor.
        hidden = output[0] if isinstance(output, tuple) else output

        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"ActivationStore configured with d_model={self.d_model} but "
                f"hook captured last dim {hidden.shape[-1]}. Check that you "
                f"hooked the right module."
            )

        flat = hidden.detach().to("cpu").reshape(-1, self.d_model)
        self._buffer.append(flat)
        self._n_tokens += flat.shape[0]

    def register_on(self, module: nn.Module) -> "ActivationStore":
        """Attach the capture hook to `module`. Call `remove()` when done."""
        if self._handle is not None:
            raise RuntimeError("ActivationStore is already registered on a module.")
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def is_full(self) -> bool:
        return self._n_tokens >= self.max_tokens

    def __len__(self) -> int:
        return self._n_tokens

    def build_dataset(self) -> torch.Tensor:
        """Concatenate all captured activations into one (N, d_model) tensor."""
        if not self._buffer:
            raise RuntimeError("No activations captured; run the model with the hook registered first.")
        dataset = torch.cat(self._buffer, dim=0)
        if dataset.shape[0] > self.max_tokens:
            dataset = dataset[: self.max_tokens]
        return dataset.to(self.device)

    def clear(self) -> None:
        self._buffer = []
        self._n_tokens = 0


def get_gpt2_block(model, layer: int) -> nn.Module:
    """Return the transformer block module at `layer` for a HF GPT-2 model.

    Works with both `GPT2LMHeadModel` (has a `.transformer` submodule) and a
    bare `GPT2Model`.
    """
    transformer = getattr(model, "transformer", model)
    n_layers = len(transformer.h)
    if not (0 <= layer < n_layers):
        raise IndexError(f"layer={layer} out of range for a {n_layers}-layer model.")
    return transformer.h[layer]


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    hook_module: nn.Module,
    dataloader,
    d_model: int,
    max_tokens: int = 500_000,
    device: str = "cpu",
    forward_fn=None,
) -> torch.Tensor:
    """Run `model` over `dataloader`, capturing activations from `hook_module`.

    Args:
        model: the model to run forward passes on (already on the target
            device, already in eval mode).
        hook_module: the specific submodule to hook (e.g. a GPT-2 block).
        dataloader: iterable of batches. Each batch is passed to `forward_fn`
            if provided, otherwise it is assumed to be a dict of tensors
            unpacked as `model(**batch)`, or a single input_ids tensor.
        d_model: hidden size of the residual stream, for shape validation.
        max_tokens: stop once this many tokens have been captured.
        device: device to run the model on and to move the final dataset to.
        forward_fn: optional callable(model, batch) -> None, for custom
            forward-pass logic (e.g. models that need attention masks).

    Returns:
        (n_tokens, d_model) tensor of flattened residual stream activations.
    """
    store = ActivationStore(d_model=d_model, max_tokens=max_tokens, device=device)
    store.register_on(hook_module)

    try:
        for batch in dataloader:
            if store.is_full():
                break
            if forward_fn is not None:
                forward_fn(model, batch)
            elif isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                model(batch.to(device))
    finally:
        store.remove()

    return store.build_dataset()
