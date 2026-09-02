"""
CLI entrypoint for the sparse-autoencoder-feature-map pipeline:

    1. Load a pretrained GPT-2 model and tokenizer.
    2. Hook a chosen transformer block and run the model over a text corpus
       to collect flattened residual stream activations.
    3. Train a SparseAutoencoder on those activations with reconstruction +
       L1 loss.
    4. Report sparsity / reconstruction metrics and save the trained SAE.

Example:
    python main.py --layer 6 --d-hidden 24576 --epochs 5 --l1-coefficient 3e-4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.activation_store import collect_activations, get_gpt2_block
from src.model import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on GPT-2 residual stream activations.")
    parser.add_argument("--model-name", type=str, default="gpt2", help="HuggingFace model id.")
    parser.add_argument("--layer", type=int, default=6, help="Transformer block index to hook.")
    parser.add_argument("--corpus", type=str, default="data/sample_corpus.txt", help="Path to a plaintext corpus.")
    parser.add_argument("--seq-len", type=int, default=64, help="Tokens per training sequence.")
    parser.add_argument("--batch-size", type=int, default=8, help="Sequences per forward pass during collection.")
    parser.add_argument("--max-tokens", type=int, default=200_000, help="Max activation vectors to collect.")

    parser.add_argument("--d-hidden", type=int, default=None, help="SAE dictionary size. Defaults to 8x d_model.")
    parser.add_argument("--l1-coefficient", type=float, default=3e-4, help="Sparsity penalty weight, lambda.")
    parser.add_argument("--tied-weights", action="store_true", default=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_corpus_batches(tokenizer, corpus_path: str, seq_len: int, batch_size: int):
    """Tokenize a plaintext file into fixed-length sequences and batch them.

    Returns a DataLoader yielding dicts of {"input_ids": LongTensor}, which
    is the format `collect_activations` expects when `forward_fn` is None.
    """
    text = Path(corpus_path).read_text(encoding="utf-8")
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]

    n_sequences = token_ids.shape[0] // seq_len
    if n_sequences == 0:
        raise ValueError(
            f"Corpus at {corpus_path} produced only {token_ids.shape[0]} tokens, "
            f"which is fewer than --seq-len={seq_len}. Use a longer corpus or a shorter seq-len."
        )
    token_ids = token_ids[: n_sequences * seq_len].view(n_sequences, seq_len)

    dataset = TensorDataset(token_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: {"input_ids": torch.stack([b[0] for b in batch])},
    )


def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> list[dict[str, float]]:
    """Train the SAE on a fixed dataset of activations. Returns per-epoch metrics."""
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        running = {"loss": 0.0, "recon_loss": 0.0, "l1_loss": 0.0, "l0": 0.0, "fvu": 0.0}
        n_batches = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = sae(batch)
            out["loss"].backward()

            sae.remove_parallel_gradient_component()
            optimizer.step()
            sae.normalize_decoder_weights()

            for key in running:
                running[key] += out[key].item()
            n_batches += 1
            progress.set_postfix(loss=out["loss"].item(), l0=out["l0"].item())

        epoch_metrics = {key: value / n_batches for key, value in running.items()}
        history.append(epoch_metrics)
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"loss={epoch_metrics['loss']:.4f}  "
            f"recon={epoch_metrics['recon_loss']:.4f}  "
            f"l1={epoch_metrics['l1_loss']:.4f}  "
            f"l0={epoch_metrics['l0']:.1f}  "
            f"fvu={epoch_metrics['fvu']:.4f}"
        )
    return history


def report_feature_sparsity(sae: SparseAutoencoder, activations: torch.Tensor, device: str) -> dict[str, float]:
    """Compute dictionary-level sparsity statistics on the full dataset."""
    sae.eval()
    densities = []
    with torch.no_grad():
        for start in range(0, activations.shape[0], 4096):
            chunk = activations[start : start + 4096].to(device)
            f = sae.encode(chunk)
            densities.append(sae.feature_density(f).cpu())
    density = torch.stack(densities).mean(dim=0)

    dead_features = (density == 0).sum().item()
    return {
        "n_features": sae.d_hidden,
        "dead_features": dead_features,
        "dead_feature_frac": dead_features / sae.d_hidden,
        "mean_feature_density": density.mean().item(),
        "median_feature_density": density.median().item(),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"loading {args.model_name} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    d_model = model.config.n_embd
    d_hidden = args.d_hidden or d_model * 8

    print(f"collecting activations from layer {args.layer} (d_model={d_model}) ...")
    dataloader = load_corpus_batches(tokenizer, args.corpus, args.seq_len, args.batch_size)
    hook_module = get_gpt2_block(model, args.layer)
    activations = collect_activations(
        model=model,
        hook_module=hook_module,
        dataloader=dataloader,
        d_model=d_model,
        max_tokens=args.max_tokens,
        device=args.device,
    )
    print(f"collected {activations.shape[0]} activation vectors of dim {activations.shape[1]}")

    sae = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        l1_coefficient=args.l1_coefficient,
        tied_weights=args.tied_weights,
    )
    print(f"training SAE: d_model={d_model} -> d_hidden={d_hidden} (expansion {d_hidden / d_model:.1f}x)")

    start = time.time()
    train_sae(
        sae=sae,
        activations=activations,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(f"training took {time.time() - start:.1f}s")

    sparsity_report = report_feature_sparsity(sae, activations, args.device)
    print("\nfeature sparsity report:")
    for key, value in sparsity_report.items():
        print(f"  {key}: {value}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"sae_layer{args.layer}.pt"
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": d_model,
            "d_hidden": d_hidden,
            "l1_coefficient": args.l1_coefficient,
            "layer": args.layer,
            "sparsity_report": sparsity_report,
        },
        checkpoint_path,
    )
    print(f"\nsaved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
