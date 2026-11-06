"""Comprehensive evaluation suite assessing strategy accuracy, baseline comparisons, slices, and safety metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from brain.config import get_preset
from brain.model import RecoverBrainModel
from brain.sandbox import resolve_brain_path
from brain.tokenizer import BrainTokenizer, BOS_ID, SEP_ID, EOS_ID

VALID_STRATEGIES = [
    "ESCALATE",
    "RESET_SERVICE",
    "RESTORE_FILE",
    "RETRY",
    "SANITIZE_DATA",
    "ROLLBACK_REPLAY",
    "PURGE_TEMP",
]
RECOVERY_STRATEGIES = [
    "RESET_SERVICE",
    "RESTORE_FILE",
    "RETRY",
    "SANITIZE_DATA",
    "ROLLBACK_REPLAY",
    "PURGE_TEMP",
]


def parse_model_proposal(raw_text: str) -> tuple[str, str]:
    """Parse generated text strictly into the closed strategy vocabulary with safety fallback to ESCALATE."""
    clean = raw_text.strip()
    tokens = clean.split("|")
    first_part = tokens[0].strip()

    strategy = "ESCALATE"
    for word in first_part.split():
        candidate = word.strip().upper()
        if candidate in VALID_STRATEGIES:
            strategy = candidate
            break

    rationale = ""
    if len(tokens) > 1 and "RATIONALE:" in tokens[1]:
        rationale = tokens[1].split("RATIONALE:")[-1].strip()
    else:
        rationale = clean

    return strategy, rationale


def evaluate_split(
    model: nn.Module,
    tokenizer: BrainTokenizer,
    split_path: Path,
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Evaluate accuracy, confusion matrix, baseline metrics, confidence calibration, and slices."""
    model.eval()
    if not split_path.exists():
        return {"error": f"File {split_path} not found"}

    with split_path.open("r", encoding="utf-8") as f:
        if max_samples is not None:
            lines = [f.readline() for _ in range(max_samples)]
        else:
            lines = f.readlines()

    lines = [l for l in lines if l.strip()]
    total = len(lines)
    if total == 0:
        return {"error": "Empty split file"}

    ground_truth_counts = Counter()
    exact_matches = 0
    confusion_matrix: dict[str, dict[str, int]] = {
        s: {s_pred: 0 for s_pred in VALID_STRATEGIES} for s in VALID_STRATEGIES
    }
    disagreements_by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_tokens_generated = 0
    total_inference_time = 0.0

    unnecessary_escalations = 0
    recoverable_ground_truths = 0

    # Slice tracking
    slice_counts = Counter()
    slice_correct = Counter()

    # Held-out family metrics (for DISK_QUOTA / PURGE_TEMP)
    held_out_correct = 0
    held_out_abstain = 0
    held_out_unsafe_wrong = 0
    is_held_out_split = "held_out" in split_path.name.lower() or "disk_quota" in split_path.name.lower()

    # Confidence calibration tracking
    confidences: list[float] = []
    correct_flags: list[bool] = []
    proposed_strategies: list[str] = []
    ground_truths: list[str] = []

    with torch.no_grad():
        for line in lines:
            data = json.loads(line)
            prompt = data["prompt"]
            ground_truth = data["strategy"]
            ground_truth_counts[ground_truth] += 1
            ground_truths.append(ground_truth)

            if ground_truth in RECOVERY_STRATEGIES:
                recoverable_ground_truths += 1

            prompt_tokens = [BOS_ID] + tokenizer.encode(prompt)[:250] + [SEP_ID]

            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0

            new_tokens = gen_tokens[len(prompt_tokens) :]
            total_tokens_generated += len(new_tokens)
            total_inference_time += elapsed

            gen_text = tokenizer.decode(new_tokens)
            proposed_strategy, _ = parse_model_proposal(gen_text)
            proposed_strategies.append(proposed_strategy)
            confidences.append(confidence)

            confusion_matrix[ground_truth][proposed_strategy] += 1
            is_correct = (proposed_strategy == ground_truth)
            correct_flags.append(is_correct)

            if is_correct:
                exact_matches += 1
            else:
                disagreements_by_type[ground_truth][proposed_strategy] += 1
                if ground_truth in RECOVERY_STRATEGIES and proposed_strategy == "ESCALATE":
                    unnecessary_escalations += 1

            # Slice tracking
            if data.get("is_history_dependent", False):
                slice_counts["history_dependent"] += 1
                if is_correct:
                    slice_correct["history_dependent"] += 1
            if data.get("is_ambiguous", False):
                slice_counts["ambiguous"] += 1
                if is_correct:
                    slice_correct["ambiguous"] += 1
            if data.get("is_conflicting", False):
                slice_counts["conflicting"] += 1
                if is_correct:
                    slice_correct["conflicting"] += 1

            # Held-out family analysis
            if is_held_out_split:
                if proposed_strategy == "PURGE_TEMP":
                    held_out_correct += 1
                elif proposed_strategy == "ESCALATE":
                    held_out_abstain += 1
                else:
                    held_out_unsafe_wrong += 1

    # Overall Accuracy
    accuracy = exact_matches / total

    # Majority-Class Baseline
    majority_class, majority_count = ground_truth_counts.most_common(1)[0]
    majority_baseline = majority_count / total
    beats_majority_baseline = accuracy > majority_baseline
    baseline_margin = accuracy - majority_baseline

    # Per-Class Precision, Recall, F1
    per_class_metrics: dict[str, dict[str, float]] = {}
    f1_list = []
    recall_list = []

    active_classes = [s for s in VALID_STRATEGIES if ground_truth_counts[s] > 0 or any(confusion_matrix[other][s] > 0 for other in VALID_STRATEGIES)]
    for s in active_classes:
        tp = confusion_matrix[s][s]
        fp = sum(confusion_matrix[other][s] for other in VALID_STRATEGIES if other != s)
        fn = sum(confusion_matrix[s][other] for other in VALID_STRATEGIES if other != s)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[s] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": ground_truth_counts[s],
        }
        f1_list.append(f1)
        recall_list.append(rec)

    macro_f1 = sum(f1_list) / len(active_classes) if active_classes else 0.0
    balanced_accuracy = sum(recall_list) / len(active_classes) if active_classes else 0.0

    # Disagreement & Escalation Rates
    total_disagreements = total - exact_matches
    disagreement_rate = total_disagreements / total
    unnecessary_escalate_rate = (
        (unnecessary_escalations / recoverable_ground_truths)
        if recoverable_ground_truths > 0
        else 0.0
    )

    tok_per_sec = (
        (total_tokens_generated / total_inference_time) if total_inference_time > 0 else 0.0
    )

    # Slice results
    slice_metrics = {}
    for sl_name, cnt in slice_counts.items():
        corr = slice_correct[sl_name]
        slice_metrics[sl_name] = {
            "count": cnt,
            "correct": corr,
            "accuracy": corr / cnt if cnt > 0 else 0.0,
        }

    # Confidence Calibration Bins
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
    calibration = []
    for low, high in bins:
        in_bin = [c for c, conf in zip(correct_flags, confidences) if low <= conf < high]
        bin_cnt = len(in_bin)
        bin_acc = sum(in_bin) / bin_cnt if bin_cnt > 0 else 0.0
        bin_confs = [conf for conf in confidences if low <= conf < high]
        bin_avg_conf = sum(bin_confs) / bin_cnt if bin_cnt > 0 else 0.0
        calibration.append({
            "range": f"[{low:.1f}, {min(1.0, high):.1f})",
            "samples": bin_cnt,
            "accuracy": bin_acc,
            "avg_confidence": bin_avg_conf,
        })

    res: dict[str, Any] = {
        "samples_evaluated": total,
        "exact_matches": exact_matches,
        "accuracy": accuracy,
        "majority_class": majority_class,
        "majority_baseline": majority_baseline,
        "beats_majority_baseline": beats_majority_baseline,
        "baseline_margin": baseline_margin,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class_metrics,
        "slices": slice_metrics,
        "calibration": calibration,
        "disagreement_rate": disagreement_rate,
        "disagreements_by_type": {k: dict(v) for k, v in disagreements_by_type.items()},
        "unnecessary_escalations": unnecessary_escalations,
        "recoverable_ground_truths": recoverable_ground_truths,
        "unnecessary_escalate_rate": unnecessary_escalate_rate,
        "tokens_per_sec": tok_per_sec,
        "confusion_matrix": confusion_matrix,
    }

    if is_held_out_split:
        res["held_out"] = {
            "total": total,
            "correct_purge_temp": held_out_correct,
            "correct_rate": held_out_correct / total,
            "safe_abstain_escalate": held_out_abstain,
            "safe_abstain_rate": held_out_abstain / total,
            "unsafe_wrong_strategy": held_out_unsafe_wrong,
            "unsafe_wrong_rate": held_out_unsafe_wrong / total,
        }

    return res


def run_threshold_sweep(
    model: nn.Module,
    tokenizer: BrainTokenizer,
    val_path: Path,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate policy behavior across escalate confidence thresholds."""
    if thresholds is None:
        thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    if not val_path.exists():
        return []

    with val_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    from brain.baselines import StatelessLookupBaseline
    stateless = StatelessLookupBaseline()

    # Pre-generate predictions and confidences
    model.eval()
    records = []
    with torch.no_grad():
        for ep in lines:
            prompt = ep["prompt"]
            prompt_tokens = [BOS_ID] + tokenizer.encode(prompt)[:250] + [SEP_ID]
            inp = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            logits, _ = model(inp)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            pred_tok = int(torch.argmax(logits[0, -1, :]).item())
            conf = float(probs[pred_tok].item())

            gen = model.generate(prompt_tokens=prompt_tokens, max_new_tokens=8, eos_token_id=EOS_ID)
            gen_text = tokenizer.decode(gen[len(prompt_tokens):])
            prop, _ = parse_model_proposal(gen_text)
            records.append({
                "episode": ep,
                "model_prop": prop,
                "confidence": conf,
                "stateless_prop": stateless.predict(ep),
                "ground_truth": ep["strategy"],
            })

    sweep_results = []
    for thresh in thresholds:
        applied_decisions = []
        for r in records:
            prop = r["model_prop"]
            if prop == "ESCALATE":
                if r["confidence"] >= thresh:
                    dec = "ESCALATE"
                else:
                    dec = r["stateless_prop"]
            else:
                dec = prop
            applied_decisions.append(dec)

        # Compute metrics
        n = len(records)
        acc = sum(1 for d, r in zip(applied_decisions, records) if d == r["ground_truth"]) / n
        rec_gt = [r for r in records if r["ground_truth"] in RECOVERY_STRATEGIES]
        unnec_esc = sum(1 for d, r in zip(applied_decisions, records) if r["ground_truth"] in RECOVERY_STRATEGIES and d == "ESCALATE")
        unnec_rate = unnec_esc / len(rec_gt) if rec_gt else 0.0

        unsafe_actions = 0
        for d, r in zip(applied_decisions, records):
            if r["ground_truth"] == "ESCALATE" and d in RECOVERY_STRATEGIES:
                unsafe_actions += 1
        unsafe_rate = unsafe_actions / sum(1 for r in records if r["ground_truth"] == "ESCALATE")

        sweep_results.append({
            "threshold": thresh,
            "accuracy": acc,
            "unnecessary_escalate_rate": unnec_rate,
            "unsafe_action_rate": unsafe_rate,
        })

    return sweep_results


def run_evaluation(
    preset_name: str = "tiny",
    checkpoint_name: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Execute full evaluation across val, test, novel, held_out, and slice benchmarks."""
    print("=" * 70)
    print(f" RecoverBrain Evaluation Suite: Preset '{preset_name.upper()}'")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = get_preset(preset_name)
    model = RecoverBrainModel(config).to(device)

    ckpt_dir = resolve_brain_path("checkpoints")
    if checkpoint_name:
        ckpt_path = ckpt_dir / checkpoint_name
    else:
        candidates = [
            ckpt_dir / f"checkpoint_{preset_name}_best.pt",
            ckpt_dir / f"checkpoint_{preset_name}_final.pt",
            ckpt_dir / f"checkpoint_{preset_name}_latest.pt",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint available for '{preset_name}' in {ckpt_dir}")

    print(f"Loading checkpoint weights from: {ckpt_path.name}")
    from brain.config import ModelConfig
    try:
        torch.serialization.add_safe_globals([ModelConfig])
    except Exception:
        pass
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    tokenizer = BrainTokenizer.load("data/tokenizer.json")
    data_dir = resolve_brain_path("data")

    splits_to_eval = ["val", "test", "novel", "held_out_disk_quota"]
    results: dict[str, Any] = {"splits": {}}

    for split in splits_to_eval:
        split_path = data_dir / f"{split}.jsonl"
        if not split_path.exists():
            continue
        print(f"\nEvaluating split: '{split}' (FULL split)...")
        eval_res = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            split_path=split_path,
            device=device,
            max_samples=max_samples,
        )
        results["splits"][split] = eval_res

        print(f"  * Total Samples Evaluated:       {eval_res['samples_evaluated']:,}")
        print(f"  * Accuracy:                      {eval_res['accuracy']*100:.2f}%")
        print(f"  * Majority-Class Baseline:       {eval_res['majority_baseline']*100:.2f}% ({eval_res['majority_class']})")
        print(f"  * Beats Majority Baseline:       {'YES' if eval_res['beats_majority_baseline'] else 'NO'} ({eval_res['baseline_margin']*100:+.2f}%)")
        print(f"  * Macro-F1:                      {eval_res['macro_f1']*100:.2f}%")
        print(f"  * Balanced Accuracy:             {eval_res['balanced_accuracy']*100:.2f}%")
        print(f"  * Unnecessary ESCALATE Rate:     {eval_res['unnecessary_escalate_rate']*100:.2f}%")

        if "slices" in eval_res and eval_res["slices"]:
            print("  Slice Performance:")
            for s_name, s_data in eval_res["slices"].items():
                print(f"    - Slice '{s_name}' (N={s_data['count']}): Acc = {s_data['accuracy']*100:.2f}%")

        if "held_out" in eval_res:
            ho = eval_res["held_out"]
            print("  Held-out Family Performance (DISK_QUOTA):")
            print(f"    - Correct (PURGE_TEMP):        {ho['correct_rate']*100:.2f}% ({ho['correct_purge_temp']}/{ho['total']})")
            print(f"    - Safe Abstain (ESCALATE):     {ho['safe_abstain_rate']*100:.2f}% ({ho['safe_abstain_escalate']}/{ho['total']})")
            print(f"    - UNSAFE Wrong Strategy:       {ho['unsafe_wrong_rate']*100:.2f}% ({ho['unsafe_wrong_strategy']}/{ho['total']})")

    # Run threshold sweep on val
    print("\nRunning Escalate Confidence Threshold Sweep on Validation Split...")
    sweep = run_threshold_sweep(model, tokenizer, data_dir / "val.jsonl", device)
    results["threshold_sweep"] = sweep
    print("  Threshold | Accuracy | Unnecessary ESC | Unsafe Action")
    print("  ------------------------------------------------------")
    for row in sweep:
        print(f"     {row['threshold']:.1f}    |  {row['accuracy']*100:5.2f}%  |     {row['unnecessary_escalate_rate']*100:5.2f}%     |     {row['unsafe_action_rate']*100:5.2f}%")

    out_file = resolve_brain_path(f"eval_{preset_name}.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved full evaluation report to: {out_file.name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RecoverBrain Model Checkpoint")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        preset_name=args.preset,
        checkpoint_name=args.checkpoint,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
