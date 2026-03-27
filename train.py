from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

from data import prepare_oasst1_for_sft, create_dataloaders
from models import ModelConfig, MiniLLM


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def global_grad_diagnostics(model: torch.nn.Module) -> tuple[float, float]:
    """
    Returns:
        grad_norm: global L2 norm of gradients
        grad_maxabs: largest absolute gradient entry
    """
    total_norm_sq = 0.0
    maxabs_overall = 0.0

    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        l2 = g.norm(2).item()
        total_norm_sq += l2 ** 2
        maxabs = g.abs().max().item()
        if maxabs > maxabs_overall:
            maxabs_overall = maxabs

    return math.sqrt(total_norm_sq), maxabs_overall


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def decode_sample(tokenizer, token_ids: torch.Tensor) -> str:
    ids = token_ids.detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf 8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: MiniLLM, loader, device: torch.device, max_batches: int | None = None) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        logits, loss = model(batch["input_ids"], batch["labels"])

        valid_mask = batch["labels"] != -100
        valid_targets = batch["labels"][valid_mask]
        valid_logits = logits[valid_mask]

        if valid_targets.numel() > 0:
            preds = valid_logits.argmax(dim=-1)
            total_correct += (preds == valid_targets).sum().item()
            total_tokens += valid_targets.numel()
            total_loss += loss.item() * valid_targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    token_acc = total_correct / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "num_eval_tokens": total_tokens,
    }


# ============================================================
# Test run
# ============================================================

def test_run(model: MiniLLM, loader, device: torch.device) -> None:
    """
    One forward and backward pass to catch shape or gradient issues.
    """
    model.train()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    logits, loss = model(batch["input_ids"], batch["labels"])

    assert logits.shape[:2] == batch["input_ids"].shape, (
        f"Logits shape {tuple(logits.shape)} is incompatible with input shape "
        f"{tuple(batch['input_ids'].shape)}"
    )

    if loss is None:
        raise RuntimeError("Loss is None during test run.")

    loss.backward()

    missing = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing gradients for: {missing}")

    model.zero_grad(set_to_none=True)


# ============================================================
# Checkpointing
# ============================================================

def checkpoint_payload(
    model: MiniLLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    run_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "run_config": run_config,
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    torch.save(payload, path)


# ============================================================
# Training
# ============================================================

def train(args) -> None:
    print("\n=== CHATBOT TRANSFORMER TRAINING ===")

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    # --------------------------------------------------------
    # Run folders
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"chatbot_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    print("\nLoading and preparing dataset...")

    bundle = prepare_oasst1_for_sft(
        model_name=args.tokenizer_name,
        max_length=args.block_size,
        train_on_all_assistant_tokens=args.train_on_all_assistant_tokens,
    )

    tokenizer = bundle["tokenizer"]

    train_loader, val_loader = create_dataloaders(
        bundle["train_tokenized"],
        bundle["val_tokenized"],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Train examples:", len(bundle["train_examples"]))
    print("Validation examples:", len(bundle["val_examples"]))
    print("Tokenizer size:", len(tokenizer))

    max_train_len = max(len(x["input_ids"]) for x in bundle["train_tokenized"][: min(1000, len(bundle["train_tokenized"]))])
    print("Max tokenized length in inspected train examples:", max_train_len)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    config = ModelConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = MiniLLM(config).to(device)

    print("\nModel:")
    print(model)
    print("Trainable parameters:", f"{count_trainable_params(model):,}")

    if args.test_run:
        test_run(model, train_loader, device)
        print("Test run OK")
        writer.close()
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # --------------------------------------------------------
    # Save static run artifacts
    # --------------------------------------------------------
    run_config = {
        "run_name": run_name,
        "seed": args.seed,
        "device": str(device),
        "tokenizer_name": args.tokenizer_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip": args.grad_clip,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
        "train_on_all_assistant_tokens": args.train_on_all_assistant_tokens,
        "snapshot_every": args.snapshot_every,
        "log_every": args.log_every,
        "eval_max_batches": args.eval_max_batches,
    }
    save_json(run_dir / "run_config.json", run_config)
    save_json(run_dir / "model_config.json", asdict(config))
    tokenizer.save_pretrained(run_dir / "tokenizer")

    history_csv = run_dir / "history.csv"
    with open(history_csv, "w", newline="", encoding="utf 8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "epoch",
            "global_step",
            "train_loss",
            "train_token_acc",
            "val_loss",
            "val_token_acc",
            "grad_norm",
            "grad_maxabs",
            "lr",
            "epoch_seconds",
        ])

    # Save a debug sample from the dataset for later inspection
    with open(sample_dir / "train_example_0.txt", "w", encoding="utf 8") as f:
        f.write("RAW EXAMPLE\n\n")
        for msg in bundle["train_examples"][0]["messages"]:
            f.write(f"[{msg['role'].upper()}]\n")
            f.write(msg["content"])
            f.write("\n\n")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss_sum = 0.0
        running_tokens = 0
        running_correct = 0

        last_grad_norm = 0.0
        last_grad_maxabs = 0.0

        for it, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)

            logits, loss = model(batch["input_ids"], batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            last_grad_norm, last_grad_maxabs = global_grad_diagnostics(model)
            optimizer.step()

            with torch.no_grad():
                valid_mask = batch["labels"] != -100
                valid_targets = batch["labels"][valid_mask]
                valid_logits = logits[valid_mask]

                if valid_targets.numel() > 0:
                    preds = valid_logits.argmax(dim=-1)
                    batch_correct = (preds == valid_targets).sum().item()
                    batch_tokens = valid_targets.numel()
                else:
                    batch_correct = 0
                    batch_tokens = 0

            running_loss_sum += loss.item() * max(batch_tokens, 1)
            running_tokens += batch_tokens
            running_correct += batch_correct

            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            writer.add_scalar("train/grad_norm", last_grad_norm, global_step)
            writer.add_scalar("train/grad_maxabs", last_grad_maxabs, global_step)
            writer.add_scalar("train/lr_iter", optimizer.param_groups[0]["lr"], global_step)

            if it % args.log_every == 0:
                avg_loss = running_loss_sum / max(running_tokens, 1)
                avg_acc = running_correct / max(running_tokens, 1)
                print(
                    f"Epoch {epoch:02d} Iter {it:04d} | "
                    f"loss={avg_loss:.4f} | token_acc={avg_acc:.4f} | "
                    f"grad_norm={last_grad_norm:.2e}"
                )

            if args.snapshot_every > 0 and global_step > 0 and global_step % args.snapshot_every == 0:
                train_metrics_snapshot = {
                    "loss": running_loss_sum / max(running_tokens, 1),
                    "token_acc": running_correct / max(running_tokens, 1),
                }
                val_metrics_snapshot = {"loss": float("nan"), "token_acc": float("nan")}
                payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    train_metrics=train_metrics_snapshot,
                    val_metrics=val_metrics_snapshot,
                    run_config=run_config,
                )
                save_checkpoint(ckpt_dir / f"snapshot_step_{global_step}.pt", payload)

            global_step += 1

        train_loss = running_loss_sum / max(running_tokens, 1)
        train_acc = running_correct / max(running_tokens, 1)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.eval_max_batches,
        )

        epoch_seconds = time.time() - epoch_start

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/token_acc_epoch", train_acc, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/token_acc", val_metrics["token_acc"], epoch)
        writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_token_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_token_acc={val_metrics['token_acc']:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

        with open(history_csv, "a", newline="", encoding="utf 8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch,
                global_step,
                train_loss,
                train_acc,
                val_metrics["loss"],
                val_metrics["token_acc"],
                last_grad_norm,
                last_grad_maxabs,
                optimizer.param_groups[0]["lr"],
                epoch_seconds,
            ])

        train_metrics_epoch = {
            "loss": train_loss,
            "token_acc": train_acc,
        }

        latest_payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            train_metrics=train_metrics_epoch,
            val_metrics=val_metrics,
            run_config=run_config,
        )

        save_checkpoint(ckpt_dir / "latest.pt", latest_payload)
        save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", latest_payload)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                train_metrics=train_metrics_epoch,
                val_metrics=val_metrics,
                run_config=run_config,
            )
            save_checkpoint(ckpt_dir / "best.pt", best_payload)

        # Save one decoded training sample batch per epoch for later debugging
        with open(sample_dir / f"epoch_{epoch:03d}_sample.txt", "w", encoding="utf 8") as f:
            batch = next(iter(train_loader))
            f.write("INPUT IDS DECODED\n\n")
            f.write(decode_sample(tokenizer, batch["input_ids"][0]))
            f.write("\n\nLABEL IDS\n\n")
            f.write(str(batch["labels"][0].tolist()))

    total_seconds = time.time() - training_start
    print(f"\nTraining complete in {total_seconds:.1f}s")
    print(f"Run folder: {run_dir}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    writer.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small chatbot transformer")

    # Run settings
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./runs/chatbot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_run", action="store_true")

    # Data
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--train_on_all_assistant_tokens", action="store_true")

    # Model
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Optimization
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging and checkpoints
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--snapshot_every", type=int, default=500)
    parser.add_argument("--eval_max_batches", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
