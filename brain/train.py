"""Production-grade PyTorch training engine optimized for RTX 5050 (8GB Blackwell, sm_120).

Key features:
  - --loss-on-target-only (default True): Computes loss exclusively on strategy + rationale tokens.
  - Logs target-token loss separately.
  - Early stopping and checkpointing tracked on Validation Macro-F1.
  - Constant token budget across presets for rigorous comparisons.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from brain.config import ModelConfig, get_preset, count_parameters
from brain.evaluate import parse_model_proposal
from brain.model import RecoverBrainModel
from brain.sandbox import resolve_brain_path
from brain.tokenizer import BrainTokenizer, PAD_ID, SEP_ID, BOS_ID, EOS_ID

VALID_STRATEGIES = [
    "RETRY",
    "RESET_SERVICE",
    "RESTORE_FILE",
    "SANITIZE_DATA",
    "ROLLBACK_REPLAY",
    "ESCALATE",
]


def find_safe_batch_size(
    model: nn.Module,
    seq_len: int,
    device: torch.device,
    target_effective_batch: int = 64,
) -> tuple[int, int]:
    """Empirically probe largest micro-batch size fitting within safe VRAM budget (~85% max)."""
    print("\n[Auto Batch Finder] Probing optimal micro-batch size on GPU...")
    model.train()

    candidate_sizes = [64, 32, 16, 8, 4, 2, 1]
    safe_batch = 1

    for b in candidate_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy_x = torch.randint(0, model.config.vocab_size, (b, seq_len), device=device)
            dummy_y = torch.randint(0, model.config.vocab_size, (b, seq_len), device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(dummy_x, labels=dummy_y)

            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)

            free_mem, total_mem = torch.cuda.mem_get_info(device)
            peak_alloc = torch.cuda.max_memory_allocated(device)

            if peak_alloc / total_mem < 0.85:
                safe_batch = b
                print(f"  * Batch size {b:2d} fits cleanly: peak VRAM={peak_alloc/(1024**3):.2f} GB")
                break
            else:
                print(f"  * Batch size {b:2d} exceeds 85% threshold, backing off...")
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"  * Batch size {b:2d} encountered CUDA OOM, backing off...")
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

    grad_accum = max(1, target_effective_batch // safe_batch)
    print(
        f"[Auto Batch Finder] Selected: micro_batch={safe_batch}, grad_accum={grad_accum} "
        f"(effective batch = {safe_batch * grad_accum})\n"
    )
    return safe_batch, grad_accum


def get_gpu_telemetry(device_id: int = 0) -> tuple[int | None, int | None, int | None]:
    """Retrieve temperature, core utilization, and power via NVML if available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Watts
        return temp, util, power
    except Exception:
        return None, None, None


def evaluate_val_macro_f1(
    model: nn.Module,
    val_data: np.ndarray,
    tokenizer: BrainTokenizer,
    device: torch.device,
    max_samples: int = 500,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate Validation Macro-F1 and Loss by probing the first predicted strategy token."""
    model.eval()
    eval_data = val_data[:max_samples]
    n = len(eval_data)
    batch_size = 32

    # Map strategy strings to token IDs for ultra-fast vectorised metric computation
    strategy_token_map = {}
    for s in VALID_STRATEGIES:
        enc = tokenizer.encode(s)
        if enc:
            strategy_token_map[s] = enc[0]

    token_to_strategy = {v: k for k, v in strategy_token_map.items()}

    tp = Counter()
    fp = Counter()
    fn = Counter()
    total_loss = 0.0
    num_loss_batches = 0

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = torch.from_numpy(eval_data[start_idx:end_idx].astype(np.int64)).to(
                device, non_blocking=True
            )

            # Compute masked validation loss
            labels = batch.clone()
            labels[labels == PAD_ID] = -100
            for i in range(batch.size(0)):
                sep_idx = (batch[i] == SEP_ID).nonzero(as_tuple=True)[0]
                if len(sep_idx) > 0:
                    labels[i, : sep_idx[0].item() + 1] = -100

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(batch, labels=labels)

            total_loss += loss.item()
            num_loss_batches += 1

            # Determine prediction and ground truth strategy for each sequence in batch
            for i in range(batch.size(0)):
                sep_idx = (batch[i] == SEP_ID).nonzero(as_tuple=True)[0]
                if len(sep_idx) == 0:
                    continue
                pos = sep_idx[0].item()

                # Ground truth token is token immediately following SEP_ID
                gt_tok = batch[i, pos + 1].item() if pos + 1 < batch.size(1) else PAD_ID
                gt_strategy = token_to_strategy.get(gt_tok, None)
                if gt_strategy is None:
                    gt_strategy, _ = parse_model_proposal(tokenizer.decode([gt_tok]))

                # Predicted token at position pos
                pred_tok = int(torch.argmax(logits[i, pos, :]).item())
                pred_strategy = token_to_strategy.get(pred_tok, None)
                if pred_strategy is None:
                    pred_strategy, _ = parse_model_proposal(tokenizer.decode([pred_tok]))

                if pred_strategy == gt_strategy:
                    tp[gt_strategy] += 1
                else:
                    fp[pred_strategy] += 1
                    fn[gt_strategy] += 1

    per_class_f1: dict[str, float] = {}
    for s in VALID_STRATEGIES:
        true_pos = tp[s]
        false_pos = fp[s]
        false_neg = fn[s]
        denom = (2 * true_pos) + false_pos + false_neg
        f1 = (2 * true_pos) / denom if denom > 0 else 0.0
        per_class_f1[s] = f1

    macro_f1 = sum(per_class_f1.values()) / len(VALID_STRATEGIES)
    mean_loss = total_loss / max(1, num_loss_batches)
    return macro_f1, mean_loss, per_class_f1


def train(
    preset_name: str = "base",
    epochs: int = 3,
    manual_batch_size: int | None = None,
    manual_grad_accum: int | None = None,
    target_effective_batch: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_hours: float = 4.0,
    target_tokens: int | None = None,
    grad_checkpoint: bool = False,
    compile_model: bool = False,
    resume: bool = False,
    loss_on_target_only: bool = True,
    eval_every_steps: int = 75,
    patience: int = 6,
    seed: int = 42,
    label_shuffle: bool = False,
    checkpoint_suffix: str = "",
) -> dict[str, Any]:
    """Execute end-to-end training loop for RecoverBrain."""
    print("=" * 70)
    print(f" RecoverBrain Training Engine: Preset '{preset_name.upper()}' | Seed: {seed}")
    print(f" Loss Mode: {'TARGET-ONLY (Observation Masked)' if loss_on_target_only else 'FULL SEQUENCE'}")
    if label_shuffle:
        print(" CONTROL: LABEL SHUFFLE ACTIVE (Targets randomly permuted)")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RecoverBrain training.")

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = get_preset(preset_name)
    model = RecoverBrainModel(config).to(device)
    param_count = count_parameters(model)
    print(f"Model Architecture: {preset_name.upper()} | Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    if grad_checkpoint:
        print("[Optimization] Gradient checkpointing ENABLED.")
        for block in model.layers:
            orig_forward = block.forward

            def make_ckpt_forward(fwd):
                return lambda x, rope: checkpoint(fwd, x, rope, use_reentrant=False)

            block.forward = make_ckpt_forward(orig_forward)

    data_dir = resolve_brain_path("data")
    train_npy = data_dir / "train.npy"
    val_npy = data_dir / "val.npy"

    if not train_npy.exists() or not val_npy.exists():
        raise FileNotFoundError(
            f"Pre-tokenized data not found in {data_dir}. Run 'python -m brain.datagen' first."
        )

    tokenizer = BrainTokenizer.load("data/tokenizer.json")
    print("Loading packed token arrays directly into system memory...")
    train_data = np.load(str(train_npy)).copy()
    val_data = np.load(str(val_npy))
    print(f"Loaded: train={len(train_data):,} samples, val={len(val_data):,} samples (seq_len={config.max_seq_len})")

    if label_shuffle:
        print("[Control] Applying independent random label assignment per example (seeded)...")
        # Extract candidate target slices
        candidate_targets = []
        for seq in train_data:
            sep_pos = np.where(seq == SEP_ID)[0]
            if len(sep_pos) > 0:
                candidate_targets.append(seq[sep_pos[0] + 1:].copy())

        rng_perm = np.random.RandomState(seed)
        # Assign an independent random target drawn uniformly with replacement per example
        for i, seq in enumerate(train_data):
            sep_pos = np.where(seq == SEP_ID)[0]
            if len(sep_pos) > 0:
                rand_idx = int(rng_perm.randint(0, len(candidate_targets)))
                train_data[i, sep_pos[0] + 1:] = candidate_targets[rand_idx].copy()

    # Determine batch size & gradient accumulation
    if manual_batch_size is not None and manual_grad_accum is not None:
        batch_size = manual_batch_size
        grad_accum = manual_grad_accum
    elif manual_batch_size is not None:
        batch_size = manual_batch_size
        grad_accum = max(1, target_effective_batch // batch_size)
    else:
        batch_size, grad_accum = find_safe_batch_size(
            model, config.max_seq_len, device, target_effective_batch
        )

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
        print("[Optimizer] Using fused CUDA AdamW.")
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        print("[Optimizer] Using standard PyTorch AdamW.")

    steps_per_epoch = len(train_data) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(10, int(total_steps * 0.05))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return learning_rate * 0.1 + 0.5 * learning_rate * 0.9 * (1.0 + math.cos(math.pi * progress))

    ckpt_dir = resolve_brain_path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    no_improve_count = 0

    start_step = 0
    if resume:
        latest_ckpt = ckpt_dir / f"checkpoint_{preset_name}_latest.pt"
        if latest_ckpt.exists():
            print(f"[Resume] Loading checkpoint from {latest_ckpt}...")
            from brain.config import ModelConfig
            try:
                torch.serialization.add_safe_globals([ModelConfig])
            except Exception:
                pass
            state = torch.load(str(latest_ckpt), map_location=device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_step = state["step"]
            best_val_macro_f1 = state.get("best_val_macro_f1", -1.0)
            print(f"[Resume] Successfully resumed from step {start_step}.")

    model.train()
    start_wall_time = time.time()
    max_seconds = max_hours * 3600
    tokens_processed = 0
    step = start_step

    train_indices = np.arange(len(train_data))
    peak_vram_recorded = 0.0
    temp_range = [100, 0]

    print("\nStarting training loop...")
    print(f"Target steps: {total_steps:,} ({epochs} epochs) | Effective batch size: {batch_size * grad_accum}")
    print(f"Checkpoint criteria: Best Validation Macro-F1 (patience={patience})\n")

    stop_training = False

    for epoch in range(epochs):
        if stop_training:
            break
        np.random.shuffle(train_indices)

        for micro_step in range(0, len(train_data) - batch_size + 1, batch_size):
            step_start_time = time.time()

            if time.time() - start_wall_time > max_seconds:
                print(f"[Budget] Reached wall-clock limit of {max_hours:.1f} hours. Stopping.")
                stop_training = True
                break

            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            batch_idx = train_indices[micro_step : micro_step + batch_size]
            batch = torch.from_numpy(train_data[batch_idx].astype(np.int64)).to(
                device, non_blocking=True
            )

            # Build label masks
            labels = batch.clone()
            labels[labels == PAD_ID] = -100

            if loss_on_target_only:
                # Mask prompt tokens up to and including SEP_ID
                for i in range(batch.size(0)):
                    sep_idx = (batch[i] == SEP_ID).nonzero(as_tuple=True)[0]
                    if len(sep_idx) > 0:
                        labels[i, : sep_idx[0].item() + 1] = -100

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, target_loss = model(batch, labels=labels)
                scaled_loss = target_loss / grad_accum

            scaled_loss.backward()
            tokens_processed += batch_size * config.max_seq_len

            if (micro_step // batch_size + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # Telemetry logging every 20 steps
                if step % 20 == 0 or step == 1:
                    step_duration = max(1e-4, time.time() - step_start_time)
                    tokens_per_sec = (batch_size * grad_accum * config.max_seq_len) / step_duration
                    alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    peak_vram_recorded = max(peak_vram_recorded, alloc_gb)

                    temp, util, power = get_gpu_telemetry(0)
                    if temp is not None:
                        temp_range[0] = min(temp_range[0], temp)
                        temp_range[1] = max(temp_range[1], temp)
                        gpu_str = f"Temp={temp}C, GPU={util}%, Pwr={power}W"
                    else:
                        gpu_str = "NVML: n/a"

                    print(
                        f"Step {step:4d}/{total_steps} (Ep {epoch+1}) | "
                        f"Target-Loss: {target_loss.item():.4f} | LR: {current_lr:.2e} | "
                        f"{tokens_per_sec:,.0f} tok/s | VRAM: {alloc_gb:.2f}GB | {gpu_str}",
                        flush=True,
                    )

                # Periodic validation and checkpointing on VAL MACRO-F1
                if step % eval_every_steps == 0 or step == total_steps:
                    val_macro_f1, val_loss, per_class_f1 = evaluate_val_macro_f1(
                        model, val_data, tokenizer, device, max_samples=500
                    )
                    f1_breakdown = " | ".join(f"{s[:3]}:{per_class_f1[s]*100:.1f}%" for s in VALID_STRATEGIES)
                    print(
                        f"\n>>> [Evaluation Step {step}] Val Macro-F1 = {val_macro_f1*100:.2f}% "
                        f"(Best: {best_val_macro_f1*100:.2f}%) | Val Loss = {val_loss:.4f}\n"
                        f"    F1 Breakdown: [{f1_breakdown}] <<<\n",
                        flush=True,
                    )

                    save_payload = {
                        "preset": preset_name,
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_macro_f1": max(best_val_macro_f1, val_macro_f1),
                        "val_loss": val_loss,
                        "config": config,
                    }

                    tag = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
                    latest_path = ckpt_dir / f"checkpoint_{preset_name}{tag}_latest.pt"
                    torch.save(save_payload, str(latest_path))

                    if val_macro_f1 > best_val_macro_f1:
                        best_val_macro_f1 = val_macro_f1
                        no_improve_count = 0
                        best_path = ckpt_dir / f"checkpoint_{preset_name}{tag}_best.pt"
                        torch.save(save_payload, str(best_path))
                        # Also save un-suffixed checkpoint if training primary model
                        if not checkpoint_suffix:
                            torch.save(save_payload, str(ckpt_dir / f"checkpoint_{preset_name}_best.pt"))
                        print(
                            f"  * New best validation Macro-F1: {best_val_macro_f1*100:.2f}% -> Saved {best_path.name}",
                            flush=True,
                        )
                    else:
                        no_improve_count += 1
                        if no_improve_count >= patience and step >= 250:
                            print(
                                f"\n[Early Stopping] Validation Macro-F1 has not improved for {patience} "
                                f"consecutive evaluation cycles. Stopping early at step {step}.\n",
                                flush=True,
                            )
                            stop_training = True
                            break

                    model.train()

            if target_tokens and tokens_processed >= target_tokens:
                print(f"[Target] Reached target tokens ({target_tokens:,}). Stopping.")
                stop_training = True
                break

    total_training_time = time.time() - start_wall_time
    final_macro_f1, final_val_loss, final_f1s = evaluate_val_macro_f1(
        model, val_data, tokenizer, device, max_samples=1000
    )

    tag = f"_{checkpoint_suffix}" if checkpoint_suffix else ""
    final_path = ckpt_dir / f"checkpoint_{preset_name}{tag}_final.pt"
    torch.save(
        {
            "preset": preset_name,
            "step": step,
            "model_state_dict": model.state_dict(),
            "final_macro_f1": final_macro_f1,
            "final_val_loss": final_val_loss,
            "config": config,
        },
        str(final_path),
    )
    if not checkpoint_suffix:
        torch.save(
            {
                "preset": preset_name,
                "step": step,
                "model_state_dict": model.state_dict(),
                "final_macro_f1": final_macro_f1,
                "final_val_loss": final_val_loss,
                "config": config,
            },
            str(ckpt_dir / f"checkpoint_{preset_name}_final.pt"),
        )

    print("\n" + "=" * 70)
    print(f" Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f" Total Tokens Processed:      {tokens_processed:,}")
    print(f" Peak VRAM Allocated:        {peak_vram_recorded:.2f} GB")
    if temp_range[0] <= temp_range[1]:
        print(f" GPU Temperature Range:      {temp_range[0]}C - {temp_range[1]}C")
    print(f" Best Validation Macro-F1:   {best_val_macro_f1*100:.2f}%")
    print(f" Final Validation Macro-F1:  {final_macro_f1*100:.2f}%")
    print(f" Checkpoints Persisted:      {final_path.name}, checkpoint_{preset_name}_best.pt")
    print("=" * 70)

    return {
        "preset": preset_name,
        "parameters": param_count,
        "steps": step,
        "training_time_sec": total_training_time,
        "tokens_processed": tokens_processed,
        "peak_vram_gb": peak_vram_recorded,
        "temp_range": temp_range,
        "best_val_macro_f1": best_val_macro_f1,
        "final_val_loss": final_val_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RecoverBrain PyTorch Training Engine")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--loss-on-target-only",
        action="store_true",
        default=True,
        help="Compute loss exclusively on target completion tokens (default: True)",
    )
    parser.add_argument(
        "--no-loss-on-target-only",
        dest="loss_on_target_only",
        action="store_false",
        help="Compute loss across the entire sequence",
    )
    parser.add_argument("--eval-every", type=int, default=75, help="Steps between validation evaluations")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience cycles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization and data shuffling")
    parser.add_argument("--label-shuffle", action="store_true", help="Control: randomly shuffle training labels")
    parser.add_argument("--checkpoint-suffix", type=str, default="", help="Suffix for checkpoint file names")
    args = parser.parse_args()

    train(
        preset_name=args.preset,
        epochs=args.epochs,
        manual_batch_size=args.batch_size,
        manual_grad_accum=args.grad_accum,
        target_effective_batch=args.effective_batch,
        learning_rate=args.lr,
        max_hours=args.max_hours,
        target_tokens=args.target_tokens,
        grad_checkpoint=args.grad_checkpoint,
        compile_model=args.compile,
        resume=args.resume,
        loss_on_target_only=args.loss_on_target_only,
        eval_every_steps=args.eval_every,
        patience=args.patience,
        seed=args.seed,
        label_shuffle=args.label_shuffle,
        checkpoint_suffix=args.checkpoint_suffix,
    )


if __name__ == "__main__":
    main()
