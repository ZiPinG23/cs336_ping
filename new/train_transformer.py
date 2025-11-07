#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.Transformer import (
    AdamW,
    TransformerLM,
    cross_entropy,
    data_loading,
    gradient_clipping,
    load_checkpoint,
    lr_cosine_scheduler,
    save_checkpoint,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the custom Transformer language model with configurable hyperparameters."
    )
    parser.add_argument("--tokenizer-pkl", type=Path, default=Path("bpe_vocab_merges.pkl"))
    parser.add_argument("--train-text", type=Path, help="Raw training text file. Required when building memmap.")
    parser.add_argument("--val-text", type=Path, help="Raw validation text file. Required when building memmap.")
    parser.add_argument("--train-memmap", type=Path, required=True, help="Binary file that backs the training memmap.")
    parser.add_argument("--val-memmap", type=Path, required=True, help="Binary file that backs the validation memmap.")
    parser.add_argument(
        "--rebuild-memmap",
        action="store_true",
        help="Force re-tokenizing the raw text even if the memmap binaries already exist.",
    )
    parser.add_argument("--storage-dtype", choices=("auto", "uint16", "uint32", "int32"), default="auto")
    parser.add_argument(
        "--memmap-chunk-chars",
        type=int,
        default=2**20,
        help="How many decoded characters to stream from the raw text at a time.",
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--max-iters", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=2_000)
    parser.add_argument("--cosine-iters", type=int, default=50_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint to resume from. Defaults to `--checkpoint-path` when it already exists.",
    )
    parser.add_argument("--log-path", type=Path, help="Optional file to append training / eval logs to.")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="If set, initialize Weights & Biases logging with the provided hyperparameters.",
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-transformer")
    parser.add_argument("--wandb-run-name", type=str, help="Override the auto-generated wandb run name.")
    parser.add_argument("--quiet", action="store_true", help="Silence stdout logging (still logs to file if provided).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_storage_dtype(vocab_size: int, storage_flag: str) -> np.dtype:
    if storage_flag == "uint16":
        return np.uint16
    if storage_flag == "uint32":
        return np.uint32
    if storage_flag == "int32":
        return np.int32
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    with open(tokenizer_path, "rb") as handle:
        packed = pickle.load(handle)
    vocab = {int(idx): bytes(token) for idx, token in packed["vocab"].items()}
    merges = [(bytes(a), bytes(b)) for a, b in packed["merges"]]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])


def _read_in_chunks(path: Path, chunk_chars: int) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as src:
        while True:
            chunk = src.read(chunk_chars)
            if not chunk:
                break
            yield chunk


def write_memmap_from_text(
    text_path: Path,
    memmap_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    chunk_chars: int,
) -> int:
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    buffer: list[int] = []
    with memmap_path.open("wb") as sink:
        for piece in tokenizer.encode_iterable(_read_in_chunks(text_path, chunk_chars)):
            if piece < 0:
                raise ValueError("Encountered token not present in the tokenizer vocabulary.")
            buffer.append(piece)
            if len(buffer) >= 1_000_000:
                np.asarray(buffer, dtype=dtype).tofile(sink)
                total_tokens += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(sink)
            total_tokens += len(buffer)
    return total_tokens


def prepare_dataset(
    memmap_path: Path,
    text_path: Path | None,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    chunk_chars: int,
    rebuild: bool,
    split_name: str,
) -> np.memmap:
    needs_build = rebuild or not memmap_path.exists()
    if needs_build:
        if text_path is None:
            raise ValueError(f"{split_name} memmap missing and no text file was provided to rebuild it.")
        print(f"[data] Building {split_name} memmap at {memmap_path} from {text_path} ...")
        start = time.time()
        total = write_memmap_from_text(text_path, memmap_path, tokenizer, dtype, chunk_chars)
        elapsed = time.time() - start
        print(f"[data] Wrote {total:,} tokens to {memmap_path} in {elapsed:.1f}s.")
    return np.memmap(memmap_path, dtype=dtype, mode="r")


def maybe_init_wandb(args: argparse.Namespace) -> None:
    if not args.use_wandb:
        return
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("wandb logging requested but wandb is not available.") from exc

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={k: v for k, v in vars(args).items() if k != "use_wandb"},
    )


def log_message(message: str, args: argparse.Namespace) -> None:
    if not args.quiet:
        print(message)
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as sink:
            sink.write(message + os.linesep)


@torch.no_grad()
def evaluate_model(
    model: TransformerLM,
    dataset: np.memmap,
    *,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = data_loading(dataset, batch_size, context_length, str(device))
        logits = model(xb).to(torch.float32)
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    precision = torch.bfloat16 if args.precision == "bf16" and device.type == "cuda" else torch.float32

    tokenizer = load_tokenizer(args.tokenizer_pkl)
    vocab_size = len(tokenizer.vocab)
    storage_dtype = choose_storage_dtype(vocab_size, args.storage_dtype)
    train_dataset = prepare_dataset(
        args.train_memmap,
        args.train_text,
        tokenizer,
        storage_dtype,
        args.memmap_chunk_chars,
        args.rebuild_memmap,
        "train",
    )
    val_dataset = prepare_dataset(
        args.val_memmap,
        args.val_text,
        tokenizer,
        storage_dtype,
        args.memmap_chunk_chars,
        args.rebuild_memmap,
        "val",
    )

    if len(train_dataset) <= args.context_length or len(val_dataset) <= args.context_length:
        raise ValueError("Datasets must be longer than the context length.")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device=device, dtype=precision)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    resume_path = args.resume_from if args.resume_from else (args.checkpoint_path if args.checkpoint_path.exists() else None)
    start_iter = 0
    if resume_path is not None and resume_path.exists():
        start_iter = load_checkpoint(resume_path, model, optimizer)
        log_message(f"[checkpoint] Resumed from {resume_path} at iteration {start_iter}.", args)

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    maybe_init_wandb(args)

    meter = {"loss": 0.0, "steps": 0}
    tokens_per_step = args.batch_size * args.context_length

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for it in range(start_iter, args.max_iters):
        lr = lr_cosine_scheduler(
            it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        raw_loss = 0.0
        for micro in range(args.grad_accumulation):
            xb, yb = data_loading(train_dataset, args.batch_size, args.context_length, str(device))
            with torch.autocast(device_type=device.type, dtype=precision, enabled=(precision != torch.float32)):
                logits = model(xb).to(torch.float32)
                loss = cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                )
                raw_loss += loss.item()
                loss = loss / args.grad_accumulation
            loss.backward()

        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        meter["loss"] += raw_loss
        meter["steps"] += 1

        if (it + 1) % args.log_interval == 0:
            avg_loss = meter["loss"] / max(meter["steps"], 1)
            ppl = math.exp(avg_loss)
            log_message(
                f"[train] iter={it+1:,} lr={lr:.2e} loss={avg_loss:.4f} ppl={ppl:.2f} tokens={tokens_per_step * (it+1):,}",
                args,
            )
            meter = {"loss": 0.0, "steps": 0}

        if (it + 1) % args.eval_interval == 0:
            val_loss = evaluate_model(
                model,
                val_dataset,
                batch_size=args.eval_batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=device,
            )
            log_message(f"[eval] iter={it+1:,} val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}", args)

        if (it + 1) % args.checkpoint_interval == 0:
            ckpt_path = args.checkpoint_path
            save_checkpoint(model, optimizer, iteration=it + 1, out=str(ckpt_path))
            log_message(f"[checkpoint] Saved checkpoint to {ckpt_path}", args)

    save_checkpoint(model, optimizer, iteration=args.max_iters, out=str(args.checkpoint_path))
    log_message(f"[done] Training completed. Final checkpoint at {args.checkpoint_path}", args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Training interrupted by user.")
