"""Comprehensive evaluation suite assessing strategy accuracy, baseline comparisons, and safety metrics."""

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

VALID_STRATEGIES = ["ESCALATE", "RESET_SERVICE", "RESTORE_FILE", "RETRY", "SANITIZE_DATA"]
RECOVERY_STRATEGIES = ["RESET_SERVICE", "RESTORE_FILE", "RETRY", "SANITIZE_DATA"]


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
    """Evaluate accuracy, confusion matrix, baseline metrics, and unnecessary escalations on full split."""
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

    with torch.no_grad():
        for line in lines:
            data = json.loads(line)
            prompt = data["prompt"]
            ground_truth = data["strategy"]
            ground_truth_counts[ground_truth] += 1

            if ground_truth in RECOVERY_STRATEGIES:
                recoverable_ground_truths += 1

            prompt_tokens = [BOS_ID] + tokenizer.encode(prompt)[:250] + [SEP_ID]

            t0 = time.perf_counter()
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

            confusion_matrix[ground_truth][proposed_strategy] += 1

            if proposed_strategy == ground_truth:
                exact_matches += 1
            else:
                disagreements_by_type[ground_truth][proposed_strategy] += 1
                if ground_truth in RECOVERY_STRATEGIES and proposed_strategy == "ESCALATE":
                    unnecessary_escalations += 1

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

    for s in VALID_STRATEGIES:
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

    macro_f1 = sum(f1_list) / len(VALID_STRATEGIES)
    balanced_accuracy = sum(recall_list) / len(VALID_STRATEGIES)

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

    return {
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
        "disagreement_rate": disagreement_rate,
        "disagreements_by_type": {k: dict(v) for k, v in disagreements_by_type.items()},
        "unnecessary_escalations": unnecessary_escalations,
        "recoverable_ground_truths": recoverable_ground_truths,
        "unnecessary_escalate_rate": unnecessary_escalate_rate,
        "tokens_per_sec": tok_per_sec,
        "confusion_matrix": confusion_matrix,
    }


def run_evaluation(
    preset_name: str = "base",
    checkpoint_name: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Execute full evaluation across val, test, and novel generalization splits."""
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

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found for preset '{preset_name}' in {ckpt_dir}")

    from brain.config import ModelConfig
    try:
        torch.serialization.add_safe_globals([ModelConfig])
    except Exception:
        pass
    print(f"Loading checkpoint weights from: {ckpt_path.name}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    tokenizer = BrainTokenizer.load("data/tokenizer.json")
    data_dir = resolve_brain_path("data")

    results: dict[str, Any] = {
        "preset": preset_name,
        "checkpoint": str(ckpt_path.name),
        "splits": {},
    }

    for split in ["val", "test", "novel"]:
        jsonl_file = data_dir / f"{split}.jsonl"
        sample_label = f"{max_samples} samples" if max_samples else "FULL split"
        print(f"\nEvaluating split: '{split}' ({sample_label})...")
        res = evaluate_split(model, tokenizer, jsonl_file, device, max_samples=max_samples)
        results["splits"][split] = res

        print(f"  * Total Samples Evaluated:       {res['samples_evaluated']:,}")
        print(f"  * Accuracy:                      {res['accuracy']*100:.2f}%")
        print(f"  * Majority-Class Baseline:       {res['majority_baseline']*100:.2f}% ({res['majority_class']})")
        verdict = f"YES (+{res['baseline_margin']*100:.2f}%)" if res['beats_majority_baseline'] else f"NO ({res['baseline_margin']*100:.2f}%)"
        print(f"  * Beats Majority Baseline:       {verdict}")
        print(f"  * Macro-F1:                      {res['macro_f1']*100:.2f}%")
        print(f"  * Balanced Accuracy:             {res['balanced_accuracy']*100:.2f}%")
        print(f"  * Disagreement Rate vs Mapping:  {res['disagreement_rate']*100:.2f}%")
        print(f"  * Unnecessary ESCALATE Rate:     {res['unnecessary_escalate_rate']*100:.2f}% ({res['unnecessary_escalations']}/{res['recoverable_ground_truths']})")
        print(f"  * Inference Throughput:          {res['tokens_per_sec']:.1f} tokens/s")

        print("\n  Per-Class Performance:")
        header_pc = f"    {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
        print(header_pc)
        print("    " + "-" * (len(header_pc) - 4))
        for cls_name in VALID_STRATEGIES:
            m = res["per_class"][cls_name]
            print(f"    {cls_name:<16} {m['precision']*100:9.2f}% {m['recall']*100:9.2f}% {m['f1']*100:9.2f}% {m['support']:10d}")

        print("\n  Confusion Matrix (Row=True, Col=Pred):")
        all_strategies = VALID_STRATEGIES
        cm_header = f"    {'True \\ Pred':<15}" + "".join(f"{s[:10]:>12}" for s in all_strategies)
        print(cm_header)
        print("    " + "-" * (len(cm_header) - 4))
        for true_s in all_strategies:
            row = f"    {true_s:<15}"
            for pred_s in all_strategies:
                cnt = res["confusion_matrix"][true_s][pred_s]
                row += f"{cnt:>12d}"
            print(row)

    eval_file = resolve_brain_path(f"eval_{preset_name}.json")
    eval_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f" Full evaluation report saved to: {eval_file.name}")
    print("=" * 70)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="RecoverBrain Rigorous Evaluation Engine")
    parser.add_argument("--preset", type=str, default="base", choices=["tiny", "small", "base"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None, help="Samples to evaluate (default: None = Full split)")
    args = parser.parse_args()

    run_evaluation(preset_name=args.preset, checkpoint_name=args.checkpoint, max_samples=args.samples)


if __name__ == "__main__":
    main()
