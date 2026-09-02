# sparse-autoencoder-feature-map

Extract residual stream activations from GPT-2 and train a sparse autoencoder
(SAE) to decompose them into a dictionary of interpretable, (mostly) monosemantic
features.

This is a minimal, dependency-light reference implementation of the SAE
dictionary-learning approach to mechanistic interpretability described in
Anthropic's *Towards Monosemanticity* (Bricken et al., 2023) and Cunningham et
al.'s *Sparse Autoencoders Find Highly Interpretable Features in Language
Models* (2023). It hooks a single GPT-2 transformer block, collects residual
stream activations over a text corpus, and trains an overcomplete,
L1-regularized autoencoder on top of them.

## Motivation

Individual neurons in a transformer's residual stream are frequently
**polysemantic** — a single neuron may fire for base64 strings, French text,
and legal boilerplate simultaneously. The leading hypothesis for this is
**superposition**: the model represents more features than it has neurons by
encoding them as (near-)orthogonal directions in activation space and
tolerating a small amount of interference between them.

A sparse autoencoder tries to undo this by learning an *overcomplete* basis
(`d_hidden >> d_model`) in which each learned direction is used sparsely. If
superposition is roughly linear, a sufficiently wide and sufficiently sparse
dictionary should recover directions that correspond to individual, causally
meaningful features rather than to entangled combinations of them.

## Math formulation

Let `x ∈ R^d_model` be a residual stream activation vector at a fixed layer
and token position. The SAE learns:

```
f     = ReLU(W_enc (x - b_dec) + b_enc)        f ∈ R^d_hidden,  f ≥ 0
x_hat = W_dec f + b_dec                        x_hat ∈ R^d_model
```

and is trained to minimize

```
L_SAE = ‖x − x_hat‖²₂  +  λ ‖f‖₁
```

where the first term is the per-token reconstruction error and the second is
an L1 penalty on the feature activations, weighted by `λ` (`l1_coefficient`).
The L1 term is what induces sparsity: it makes the optimizer prefer solutions
where most entries of `f` are exactly zero for any given input, rather than
solutions that spread reconstruction responsibility thinly across every
feature.

**Decoder unit-norm constraint.** Because `L1` scales `f` while the
reconstruction only cares about `W_dec f`, gradient descent can trivially
"cheat" on the sparsity penalty by shrinking `‖W_dec‖` and inflating `f`
proportionally, with zero effect on reconstruction quality. To prevent this,
`W_dec` rows are renormalized to unit L2 norm after every optimizer step
(`normalize_decoder_weights`), and the component of the gradient parallel to
each `W_dec` row is projected out before the step
(`remove_parallel_gradient_component`), so gradient descent only moves each
dictionary direction along the unit sphere.

**Evaluation metrics** reported by this repo:

| Metric | Meaning |
|---|---|
| `L0` | average number of features active (non-zero) per token — the *effective sparsity* |
| `FVU` | fraction of variance unexplained, `Var(x − x_hat) / Var(x)` — reconstruction quality |
| feature density | fraction of tokens on which each individual feature fires |
| dead features | features with zero density over the evaluation set — wasted dictionary capacity |

## Architecture

```
                     GPT-2 (frozen, eval mode)
   tokens ─────▶ [emb] ─▶ [block 0] ─▶ ... ─▶ [block L] ─▶ ... ─▶ [block 11]
                                            │
                                            │  forward hook captures
                                            │  residual stream output
                                            │  (batch, seq, d_model)
                                            ▼
                                  ┌───────────────────┐
                                  │  ActivationStore   │  flatten to
                                  │  (src/activation_  │  (n_tokens, d_model)
                                  │      store.py)      │
                                  └─────────┬──────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────┐
                        │       SparseAutoencoder            │
                        │       (src/model.py)               │
                        │                                     │
                        │   x (d_model)                       │
                        │     │                                │
                        │     ▼  W_enc, b_enc                  │
                        │   [ Linear + ReLU ]                  │
                        │     │                                │
                        │     ▼  f (d_hidden, sparse, f ≥ 0)    │
                        │     │                                │
                        │     ▼  W_dec (unit-norm rows), b_dec  │
                        │   [ Linear ]                          │
                        │     │                                │
                        │     ▼  x_hat (d_model)                │
                        │                                     │
                        │   L = ‖x - x_hat‖² + λ‖f‖₁          │
                        └───────────────────────────────────┘
                                            │
                                            ▼
                             feature sparsity / density report
                                  outputs/sae_layer{L}.pt
```

`d_hidden` is intentionally larger than `d_model` (default 8x — GPT-2 small
has `d_model = 768`, so the default dictionary has 6144 features) so the
autoencoder has room to unpack superposed directions into separate features.

## Repository structure

```
sparse-autoencoder-feature-map/
├── main.py                      # CLI: collect activations, train SAE, report metrics
├── requirements.txt
├── data/
│   └── sample_corpus.txt        # small bundled corpus for a quick offline-friendly demo
├── src/
│   ├── model.py                 # SparseAutoencoder (encoder/decoder, L1 loss, dict-learning utils)
│   └── activation_store.py      # forward-hook activation capture
├── tests/
│   └── test_pipeline_offline.py # end-to-end sanity checks against a toy model (no GPT-2 download needed)
└── outputs/                     # trained SAE checkpoints land here
```

## Installation

```bash
git clone https://github.com/<your-username>/sparse-autoencoder-feature-map.git
cd sparse-autoencoder-feature-map
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train an SAE on layer 6 of GPT-2 small using the bundled sample corpus:

```bash
python main.py --layer 6 --epochs 5
```

Full options:

```bash
python main.py \
  --model-name gpt2 \
  --layer 6 \                 # which transformer block to hook (0-indexed)
  --corpus data/sample_corpus.txt \
  --seq-len 64 \               # tokens per training sequence during collection
  --batch-size 8 \              # sequences per forward pass during collection
  --max-tokens 200000 \         # cap on collected activation vectors
  --d-hidden 6144 \             # dictionary size (defaults to 8x d_model)
  --l1-coefficient 3e-4 \       # sparsity penalty weight
  --epochs 5 \
  --train-batch-size 1024 \
  --lr 1e-3 \
  --device cuda                 # or cpu
```

To point the pipeline at your own text, swap in a larger `--corpus` file
(e.g. a WikiText or OpenWebText dump) — the tokenizer, hook, and training
loop don't need any other changes.

On completion, `main.py` prints a sparsity report:

```
feature sparsity report:
  n_features: 6144
  dead_features: 412
  dead_feature_frac: 0.067
  mean_feature_density: 0.014
  median_feature_density: 0.003
```

and saves a checkpoint to `outputs/sae_layer6.pt` containing the SAE's
`state_dict`, its hyperparameters, and the sparsity report.

## Running the offline sanity tests

`tests/test_pipeline_offline.py` exercises the activation hook, the SAE
forward/backward pass, the decoder-normalization and gradient-projection
utilities, and a short end-to-end training loop against a small hand-built
model — no GPT-2 download or GPU required:

```bash
python tests/test_pipeline_offline.py
```

## Extending this repo

- **Resampling dead features**: periodically reinitialize dictionary
  directions with zero density on activations with high reconstruction
  error, per Bricken et al.'s resampling procedure.
- **Auto-interpretability**: for each learned feature, pull the top-activating
  tokens/contexts from the corpus and (optionally) have a language model
  summarize the pattern.
- **Multi-layer sweep**: loop `main.py` over `--layer` to compare
  reconstruction quality and dead-feature rate across depth.
- **Ghost grads / other dead-feature mitigations**, **JumpReLU or TopK
  activation functions**, and **cross-layer transcoders** are natural next
  steps once this baseline is working end-to-end.

## References

- Bricken et al., *Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning*, Anthropic, 2023.
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features
  in Language Models*, 2023.
- Olshausen & Field, *Sparse Coding with an Overcomplete Basis Set: A
  Strategy Employed by V1?*, 1997.
