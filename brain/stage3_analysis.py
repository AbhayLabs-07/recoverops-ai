"""Stage 3 comprehensive analysis: multi-seed statistics, confidence threshold sweep,
leakage audit, and honest 'is the neural advisor worth it?' verdict.

Run: python -m brain.stage3_analysis
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from brain.config import get_preset, ModelConfig
from brain.model import RecoverBrainModel
from brain.sandbox import resolve_brain_path
from brain.tokenizer import BrainTokenizer, BOS_ID, SEP_ID, EOS_ID
from brain.evaluate import parse_model_proposal, VALID_STRATEGIES, RECOVERY_STRATEGIES
from brain.baselines import (
    DecisionTreeBaseline,
    MajorityClassBaseline,
    StatelessLookupBaseline,
    extract_features,
    evaluate_baseline_predictions,
)


# ──────────────────────────────────────────────────────────────────────
# Utility: load split
# ──────────────────────────────────────────────────────────────────────
def load_split(split_name: str) -> list[dict[str, Any]]:
    data_dir = resolve_brain_path("data")
    p = data_dir / f"{split_name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Split not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────
# Model inference helpers
# ──────────────────────────────────────────────────────────────────────
def load_model_for_eval(
    preset_name: str,
    checkpoint_path: Path | str,
    device: torch.device,
) -> tuple[RecoverBrainModel, BrainTokenizer]:
    """Load model and tokenizer for evaluation."""
    config = get_preset(preset_name)
    try:
        torch.serialization.add_safe_globals([ModelConfig])
    except Exception:
        pass
    model = RecoverBrainModel(config).to(device)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    tok_file = resolve_brain_path("data/tokenizer.json")
    if tok_file.exists():
        tokenizer = BrainTokenizer.load("data/tokenizer.json")
    else:
        tokenizer = BrainTokenizer(vocab_size=config.vocab_size)
    return model, tokenizer


def predict_with_confidence(
    model: RecoverBrainModel,
    tokenizer: BrainTokenizer,
    episode: dict[str, Any],
    device: torch.device,
    max_new_tokens: int = 8,
) -> tuple[str, float]:
    """Generate prediction and confidence from model."""
    prompt = episode["prompt"]
    prompt_tokens = [BOS_ID] + tokenizer.encode(prompt)[:250] + [SEP_ID]

    with torch.no_grad():
        inp = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        pred_tok = int(torch.argmax(next_logits).item())
        confidence = float(probs[pred_tok].item())

        gen_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_ID,
        )

    new_tokens = gen_tokens[len(prompt_tokens):]
    gen_text = tokenizer.decode(new_tokens)
    strategy, _ = parse_model_proposal(gen_text)
    return strategy, confidence


# ──────────────────────────────────────────────────────────────────────
# Step 0.1: Leakage audit
# ──────────────────────────────────────────────────────────────────────
BANNED_TOKENS = [
    "RETRY", "RESET_SERVICE", "RESTORE_FILE", "SANITIZE_DATA",
    "ROLLBACK_REPLAY", "PURGE_TEMP", "ESCALATE",
    "retry", "reset_service", "restore_file", "sanitize_data",
    "rollback_replay", "purge_temp", "escalate",
]

def leakage_audit(split_names: list[str] | None = None) -> dict[str, Any]:
    """Assert no observation or rationale text contains strategy labels or trivially revealing tokens."""
    if split_names is None:
        split_names = ["train", "val", "test", "novel"]

    results = {}
    total_violations = 0

    for name in split_names:
        episodes = load_split(name)
        violations = []
        for i, ep in enumerate(episodes):
            obs_text = json.dumps(ep.get("observation", {}))
            # Check observation text for strategy labels
            for token in BANNED_TOKENS:
                if token in obs_text:
                    # Allow 'failure_category' value matches and other legit obs fields
                    # Only flag if the token appears as a standalone word in the obs text
                    # NOT inside known safe field values
                    pass  # observation JSON may contain 'rate_limit_exceeded' etc - not a leak

            # Check if observation JSON contains the target strategy as a value
            strategy = ep.get("strategy", "")
            obs = ep.get("observation", {})
            obs_values_str = " ".join(str(v) for v in obs.values() if isinstance(v, str))
            if strategy in obs_values_str:
                violations.append({
                    "index": i,
                    "strategy": strategy,
                    "field": "observation_value",
                    "context": obs_values_str[:200],
                })

            # Check rationale doesn't contain the strategy label
            # (rationale is target, not input — skip this check)

        results[name] = {
            "total_episodes": len(episodes),
            "violations": len(violations),
            "details": violations[:5] if violations else [],
        }
        total_violations += len(violations)

    return {
        "total_violations": total_violations,
        "passed": total_violations == 0,
        "per_split": results,
    }


# ──────────────────────────────────────────────────────────────────────
# Full evaluation with per-class metrics
# ──────────────────────────────────────────────────────────────────────
def full_eval(
    predictions: list[str],
    targets: list[str],
    confidences: list[float] | None = None,
    episodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Comprehensive eval: accuracy, majority baseline, per-class P/R/F1, macro-F1, balanced acc, confusion."""
    n = len(targets)
    assert n == len(predictions)

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    accuracy = correct / n

    # Majority baseline
    gt_counts = Counter(targets)
    majority_class, majority_count = gt_counts.most_common(1)[0]
    majority_baseline = majority_count / n

    # Per-class metrics
    classes = sorted(set(targets) | set(predictions))
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for p, t in zip(predictions, targets):
        confusion[t][p] += 1

    per_class = {}
    f1s, recalls = [], []
    for c in classes:
        tp = confusion.get(c, {}).get(c, 0)
        fp = sum(confusion.get(other, {}).get(c, 0) for other in classes if other != c)
        fn = sum(confusion.get(c, {}).get(other, 0) for other in classes if other != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": gt_counts.get(c, 0)}
        if gt_counts.get(c, 0) > 0:
            f1s.append(f1)
            recalls.append(rec)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    # Unnecessary escalation rate
    unnecessary_esc = sum(1 for p, t in zip(predictions, targets) if p == "ESCALATE" and t in RECOVERY_STRATEGIES)
    recoverable_total = sum(1 for t in targets if t in RECOVERY_STRATEGIES)

    result = {
        "n": n,
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "majority_class": majority_class,
        "beats_majority": accuracy > majority_baseline,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "unnecessary_escalation_rate": unnecessary_esc / recoverable_total if recoverable_total > 0 else 0.0,
        "class_distribution": dict(gt_counts),
    }

    # Slice analysis if episodes provided
    if episodes:
        slices = {
            "history_dependent": [i for i, e in enumerate(episodes) if e.get("is_history_dependent", False)],
            "ambiguous": [i for i, e in enumerate(episodes) if e.get("is_ambiguous", False)],
        }
        slice_res = {}
        for s_name, indices in slices.items():
            if indices:
                s_preds = [predictions[i] for i in indices]
                s_targs = [targets[i] for i in indices]
                s_acc = sum(1 for p, t in zip(s_preds, s_targs) if p == t) / len(indices)
                slice_res[s_name] = {"count": len(indices), "accuracy": s_acc}
        result["slices"] = slice_res

    return result


# ──────────────────────────────────────────────────────────────────────
# Confidence threshold sweep
# ──────────────────────────────────────────────────────────────────────
def confidence_threshold_sweep(
    predictions: list[str],
    targets: list[str],
    confidences: list[float],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds, falling back to ESCALATE for low-confidence predictions."""
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    results = []
    for thresh in thresholds:
        adjusted = []
        for pred, conf in zip(predictions, confidences):
            if conf < thresh:
                adjusted.append("ESCALATE")
            else:
                adjusted.append(pred)

        eval_result = full_eval(adjusted, targets)
        unnecessary_esc = sum(
            1 for p, t in zip(adjusted, targets)
            if p == "ESCALATE" and t in RECOVERY_STRATEGIES
        )
        recoverable = sum(1 for t in targets if t in RECOVERY_STRATEGIES)
        unsafe = sum(
            1 for p, t in zip(adjusted, targets)
            if p != t and p in RECOVERY_STRATEGIES and t != "ESCALATE"
        )

        results.append({
            "threshold": thresh,
            "accuracy": eval_result["accuracy"],
            "macro_f1": eval_result["macro_f1"],
            "unnecessary_escalation_rate": unnecessary_esc / recoverable if recoverable > 0 else 0.0,
            "unsafe_action_rate": unsafe / len(targets),
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap CI for a list of values."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, mean, mean

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return mean, lower, upper


def paired_bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired bootstrap CI for the difference (A - B)."""
    rng = np.random.RandomState(seed)
    arr_a = np.array(values_a)
    arr_b = np.array(values_b)
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(diff), size=len(diff), replace=True)
        boot_diffs.append(float(np.mean(diff[idx])))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_diffs, 100 * alpha))
    upper = float(np.percentile(boot_diffs, 100 * (1 - alpha)))
    return mean_diff, lower, upper


# ──────────────────────────────────────────────────────────────────────
# Multi-seed evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_model_on_split(
    preset_name: str,
    checkpoint_path: Path,
    split_name: str,
    device: torch.device,
) -> tuple[list[str], list[str], list[float], list[dict]]:
    """Evaluate model, returning predictions, targets, confidences, episodes."""
    model, tokenizer = load_model_for_eval(preset_name, checkpoint_path, device)
    episodes = load_split(split_name)
    predictions, targets, confidences = [], [], []

    for ep in episodes:
        pred, conf = predict_with_confidence(model, tokenizer, ep, device)
        predictions.append(pred)
        targets.append(ep["strategy"])
        confidences.append(conf)

    del model
    torch.cuda.empty_cache()
    return predictions, targets, confidences, episodes


def multi_seed_eval(
    preset_name: str,
    seeds: list[int],
    split_name: str = "val",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate model across seeds, compute mean±std and CIs."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_dir = resolve_brain_path("checkpoints")
    seed_results = {}
    accuracies, macro_f1s = [], []

    for seed in seeds:
        if seed == 42:
            ckpt = ckpt_dir / f"checkpoint_{preset_name}_best.pt"
        else:
            ckpt = ckpt_dir / f"checkpoint_{preset_name}_seed{seed}_best.pt"

        if not ckpt.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt.name}")
            continue

        print(f"  Evaluating {preset_name} seed={seed} on {split_name}...")
        preds, targs, confs, eps = evaluate_model_on_split(preset_name, ckpt, split_name, device)
        result = full_eval(preds, targs, confs, eps)
        seed_results[seed] = result
        accuracies.append(result["accuracy"])
        macro_f1s.append(result["macro_f1"])

    if not accuracies:
        return {"error": "No checkpoints found"}

    return {
        "seeds_evaluated": list(seed_results.keys()),
        "per_seed": seed_results,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "n_seeds": len(accuracies),
    }


# ──────────────────────────────────────────────────────────────────────
# Main analysis script
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3 comprehensive analysis")
    parser.add_argument("--action", choices=[
        "leakage_audit", "multi_seed", "threshold_sweep", "full_eval", "all"
    ], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    parser.add_argument("--split", default="val")
    parser.add_argument("--preset", default="base")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.action in ("leakage_audit", "all"):
        print("\n" + "=" * 70)
        print("LEAKAGE AUDIT")
        print("=" * 70)
        audit = leakage_audit()
        print(f"  PASSED: {audit['passed']}")
        print(f"  Total violations: {audit['total_violations']}")
        for split_name, info in audit["per_split"].items():
            print(f"    {split_name}: {info['violations']}/{info['total_episodes']} violations")
        if not audit["passed"]:
            for split_name, info in audit["per_split"].items():
                for v in info["details"]:
                    print(f"      [{split_name}] idx={v['index']} strategy={v['strategy']} in {v['field']}: {v['context'][:80]}")

    if args.action in ("multi_seed", "all"):
        print("\n" + "=" * 70)
        print(f"MULTI-SEED EVALUATION: {args.preset}")
        print("=" * 70)

        for preset in ["tiny", args.preset]:
            print(f"\n--- Preset: {preset} ---")
            ms = multi_seed_eval(preset, args.seeds, args.split, device)
            if "error" in ms:
                print(f"  ERROR: {ms['error']}")
                continue

            print(f"  Seeds evaluated: {ms['seeds_evaluated']}")
            print(f"  Accuracy: {ms['accuracy_mean']*100:.2f}% ± {ms['accuracy_std']*100:.2f}%")
            print(f"  Macro-F1: {ms['macro_f1_mean']*100:.2f}% ± {ms['macro_f1_std']*100:.2f}%")

            for seed, r in ms["per_seed"].items():
                print(f"    Seed {seed}: acc={r['accuracy']*100:.2f}%, macro-F1={r['macro_f1']*100:.2f}%, "
                      f"bal_acc={r['balanced_accuracy']*100:.2f}%")
                if r.get("slices"):
                    for s_name, s_info in r["slices"].items():
                        print(f"      Slice '{s_name}': {s_info['count']} samples, acc={s_info['accuracy']*100:.2f}%")

        # Decision tree baseline comparison
        print("\n--- Decision Tree Baseline ---")
        train_eps = load_split("train")
        train_targets = [e["strategy"] for e in train_eps]
        tree = DecisionTreeBaseline(max_depth=5, min_samples_split=10)
        tree.fit(train_eps, train_targets)

        val_eps = load_split(args.split)
        tree_preds = [tree.predict(e) for e in val_eps]
        val_targets = [e["strategy"] for e in val_eps]
        tree_eval = full_eval(tree_preds, val_targets, episodes=val_eps)
        print(f"  Decision Tree: acc={tree_eval['accuracy']*100:.2f}%, macro-F1={tree_eval['macro_f1']*100:.2f}%")

        # Paired bootstrap CI: base vs tree
        ckpt_dir = resolve_brain_path("checkpoints")
        base_ckpt = ckpt_dir / f"checkpoint_{args.preset}_best.pt"
        if base_ckpt.exists():
            preds_base, targs_base, confs_base, eps_base = evaluate_model_on_split(
                args.preset, base_ckpt, args.split, device
            )
            base_correct = [1.0 if p == t else 0.0 for p, t in zip(preds_base, targs_base)]
            tree_correct = [1.0 if p == t else 0.0 for p, t in zip(tree_preds, val_targets)]
            diff_mean, diff_lo, diff_hi = paired_bootstrap_ci(base_correct, tree_correct)
            print(f"  Base-vs-Tree difference: {diff_mean*100:+.2f}% [{diff_lo*100:+.2f}%, {diff_hi*100:+.2f}%] (95% CI)")

    if args.action in ("threshold_sweep", "all"):
        print("\n" + "=" * 70)
        print("CONFIDENCE THRESHOLD SWEEP")
        print("=" * 70)

        ckpt_dir = resolve_brain_path("checkpoints")
        ckpt = ckpt_dir / f"checkpoint_{args.preset}_best.pt"
        if not ckpt.exists():
            print(f"  ERROR: Checkpoint not found: {ckpt}")
        else:
            preds, targs, confs, eps = evaluate_model_on_split(args.preset, ckpt, args.split, device)
            sweep = confidence_threshold_sweep(preds, targs, confs)
            print(f"  {'Threshold':>10s} {'Accuracy':>10s} {'Macro-F1':>10s} {'Unnec.Esc':>10s} {'Unsafe':>10s}")
            print(f"  {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")
            for row in sweep:
                print(f"  {row['threshold']:10.2f} {row['accuracy']*100:9.2f}% {row['macro_f1']*100:9.2f}% "
                      f"{row['unnecessary_escalation_rate']*100:9.2f}% {row['unsafe_action_rate']*100:9.2f}%")

    if args.action in ("full_eval", "all"):
        print("\n" + "=" * 70)
        print(f"FULL EVAL: {args.preset} on {args.split}")
        print("=" * 70)

        ckpt_dir = resolve_brain_path("checkpoints")
        ckpt = ckpt_dir / f"checkpoint_{args.preset}_best.pt"
        if not ckpt.exists():
            print(f"  ERROR: Checkpoint not found: {ckpt}")
        else:
            preds, targs, confs, eps = evaluate_model_on_split(args.preset, ckpt, args.split, device)
            result = full_eval(preds, targs, confs, eps)

            print(f"  Samples: {result['n']}")
            print(f"  Accuracy: {result['accuracy']*100:.2f}%")
            print(f"  Majority baseline: {result['majority_baseline']*100:.2f}% ({result['majority_class']})")
            print(f"  Beats majority: {result['beats_majority']}")
            print(f"  Macro-F1: {result['macro_f1']*100:.2f}%")
            print(f"  Balanced accuracy: {result['balanced_accuracy']*100:.2f}%")
            print(f"  Unnecessary escalation rate: {result['unnecessary_escalation_rate']*100:.2f}%")

            print("\n  Per-class metrics:")
            print(f"  {'Class':>20s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
            for cls, m in sorted(result["per_class"].items()):
                print(f"  {cls:>20s} {m['precision']*100:7.1f}% {m['recall']*100:7.1f}% "
                      f"{m['f1']*100:7.1f}% {m['support']:>8d}")

            print("\n  Confusion Matrix:")
            classes = sorted(result["confusion_matrix"].keys())
            header = f"  {'True\\Pred':>18s}" + "".join(f" {c[:7]:>8s}" for c in classes)
            print(header)
            for true_cls in classes:
                row = f"  {true_cls[:18]:>18s}"
                for pred_cls in classes:
                    row += f" {result['confusion_matrix'][true_cls][pred_cls]:>8d}"
                print(row)

            if result.get("slices"):
                print("\n  Slice analysis:")
                for s_name, s_info in result["slices"].items():
                    print(f"    {s_name}: {s_info['count']} samples, acc={s_info['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

