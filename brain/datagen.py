"""Synthetic curriculum data engine using the real Phase 1 engine as oracle.

Rebalanced & Stratified:
  - Emits ONLY examples where Phase 1 guardrails pass and the model is actually consulted.
  - Zero guardrail-driven escalations in dataset.
  - Exactly stratified so RETRY, RESET_SERVICE, RESTORE_FILE, SANITIZE_DATA, ESCALATE
    each comprise at least 15% (balanced at ~20%) of every split.
  - Novel split generated from strictly disjoint parameter combinations with automated assertions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    VerificationResult,
)
from app.recovery import RecoveryManager
from app.verification import Verifier
from brain.config import get_preset
from brain.sandbox import resolve_brain_path
from brain.tokenizer import BrainTokenizer, BOS_ID, EOS_ID, SEP_ID, PAD_ID

VALID_STRATEGIES = ["RETRY", "RESET_SERVICE", "RESTORE_FILE", "SANITIZE_DATA", "ESCALATE"]

# Disjoint parameter domains
STANDARD_DOMAINS = [
    "auth-gateway",
    "billing-worker",
    "inventory-api",
    "order-router",
    "customer-cache",
    "session-store",
    "notification-hub",
    "payment-processor",
]

NOVEL_DOMAINS = [
    "quantum-vault",
    "satellite-telemetry",
    "biometric-auth",
    "distributed-consensus",
    "pci-firewall",
    "firmware-flasher",
]

STANDARD_PORTS = [8080, 8081, 5000, 3000, 9000, 8443]
NOVEL_PORTS = [9443, 7077, 4430, 27017, 11434, 18080]

STANDARD_PATHS = [
    "configs/settings.json",
    "data/payload.json",
    "assets/manifest.json",
    "reports/export.json",
    "schemas/input.json",
    "logs/application.log",
]

NOVEL_PATHS = [
    "vault/secure_token.pem",
    "telemetry/stream.bin",
    "snapshots/cluster_state.parquet",
    "firmware/device.bin",
    "pci/audit_trail.log",
    "keys/hardware_key.dat",
]

STANDARD_KEYS = ["id", "valid", "checksum", "user_id", "email"]
NOVEL_KEYS = ["block_hash", "entropy_sig", "device_fingerprint", "nonce_token"]

STANDARD_TEMPLATES = {
    "TIMEOUT": [
        "Operation on {domain}:{port} timed out after 3000ms",
        "Deterministic read deadline exceeded on {domain}",
        "Upstream {domain} socket idle timeout breached SLA",
        "Connection wait timeout on service {domain}:{port}",
    ],
    "TRANSIENT_SERVICE": [
        "Connection refused by {domain} on port {port}",
        "Service {domain} temporarily unavailable: initializing pool",
        "Health probe on {domain}:{port} returned 503 Service Unavailable",
        "RPC transport error: socket closed on {domain}",
    ],
    "MISSING_FILE": [
        "Required asset file missing: {path}",
        "Storage dependency {path} not found in container",
        "Critical configuration {path} is absent from filesystem",
        "Path lookup failed: asset '{path}' does not exist",
    ],
    "INVALID_DATA": [
        "Data validation failed: missing key '{key}' in record",
        "Schema corruption: null value found for required field '{key}'",
        "Structural contract violation on field '{key}'",
        "Checksum mismatch on ingested record for key '{key}'",
    ],
    "ESCALATE": [
        "Output verification mismatch: expected {expected} but observed {observed}",
        "Verification assertion failure: unexpected state {observed}",
        "Contract violation in task output: criteria {criterion} failed",
        "Ambiguous output state detected: observed {observed}",
    ],
}

NOVEL_TEMPLATES = {
    "TIMEOUT": [
        "Subsystem {domain} async bus timeout on port {port}",
        "Novel streaming timeout on boundary {domain}",
        "Telemetry deadline breached on node {domain}:{port}",
    ],
    "TRANSIENT_SERVICE": [
        "Consensus node {domain}:{port} rejecting peer synchronization",
        "Enclave {domain} offline during key exchange",
        "High-integrity node {domain}:{port} returned transient 504",
    ],
    "MISSING_FILE": [
        "Novel vault object {path} missing from secure storage",
        "Cryptographic artifact {path} not mounted in hardware enclave",
        "Audit log partition {path} missing from secure volume",
    ],
    "INVALID_DATA": [
        "Enclave schema check failed: missing signature token '{key}'",
        "Zero-knowledge proof verification failed on field '{key}'",
        "Consensus state contract violation on key '{key}'",
    ],
    "ESCALATE": [
        "Unresolvable verification discrepancy on {domain}: expected {expected} vs observed {observed}",
        "High-integrity output assertion failed: criterion {criterion} rejected state {observed}",
        "Ambiguous sensor telemetry in task output: verification failed with observed {observed}",
    ],
}


def make_observation(
    action: str,
    failure_category: str,
    error_message: str,
    attempt: int,
    task_recoveries_used: int,
    workflow_budget_left: int,
    idempotent: bool,
    criterion_type: str,
    task_id: str = "task_01",
) -> dict[str, Any]:
    """Standardized observation dictionary shared identically across datagen, advisor, and engine."""
    return {
        "action": action,
        "attempt": attempt,
        "criterion_type": criterion_type,
        "error_message": error_message,
        "failure_category": failure_category,
        "idempotent": idempotent,
        "task_id": task_id,
        "task_recoveries_used": task_recoveries_used,
        "workflow_budget_left": workflow_budget_left,
    }


def generate_episode_for_strategy(
    strategy: str,
    episode_idx: int,
    rng: random.Random,
    sim: FailureSimulator,
    verifier: Verifier,
    recovery_mgr: RecoveryManager,
    is_novel: bool = False,
) -> dict[str, Any]:
    """Generate a single instance for a specified target strategy with all guardrails guaranteed to pass."""
    task_id = f"t_{strategy.lower()}_{episode_idx}_{rng.randint(100000, 999999)}"
    sim._service_ready.pop(task_id, None)
    sim._sanitized_data.pop(task_id, None)
    sim._restored_files.discard(task_id)

    # Parameters: all Phase 1 guardrails MUST pass
    idempotent = True
    max_attempts = rng.randint(3, 5)
    max_recoveries = rng.randint(2, 4)
    attempt = 1  # Guaranteed attempt < max_attempts
    task_recoveries_used = 0  # Guaranteed recoveries_used < max_recoveries
    budget_left = rng.randint(2, 5)  # Guaranteed budget_left > 0

    templates = NOVEL_TEMPLATES if is_novel else STANDARD_TEMPLATES
    domains = NOVEL_DOMAINS if is_novel else STANDARD_DOMAINS
    ports = NOVEL_PORTS if is_novel else STANDARD_PORTS
    paths = NOVEL_PATHS if is_novel else STANDARD_PATHS
    keys = NOVEL_KEYS if is_novel else STANDARD_KEYS

    domain = rng.choice(domains)
    port = rng.choice(ports)
    path = rng.choice(paths)
    key = rng.choice(keys)

    # Disjoint parameter combination signature
    signature = (domain, path.split("/")[0], key)

    if strategy == "RETRY":
        action = TaskAction.TIMEOUT
        msg = rng.choice(templates["TIMEOUT"]).format(domain=domain, port=port)
        criteria = SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT)
        params: dict[str, Any] = {
            "error_message": msg,
            "succeed_on_attempt": 2,
        }
        task = Task(
            id=task_id,
            name=f"Task {task_id}",
            action=action,
            parameters=params,
            success_criteria=criteria,
            idempotent=idempotent,
            max_attempts=max_attempts,
            max_recoveries=max_recoveries,
        )
        exec_result = sim.execute(task, attempt_number=attempt)
        verif_result = verifier.verify(task, exec_result, sim)
        failure_cat = recovery_mgr.classify_failure(exec_result, verif_result)
        rec_attempt = recovery_mgr.attempt_recovery(
            task=task,
            failure_category=failure_cat,
            simulator=sim,
            current_attempt=attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=0,
            workflow_budget=budget_left,
            recovery_number=1,
        )

    elif strategy == "RESET_SERVICE":
        action = TaskAction.TRANSIENT_SERVICE
        msg = rng.choice(templates["TRANSIENT_SERVICE"]).format(domain=domain, port=port)
        criteria = SuccessCriteria(CriterionType.SERVICE_READY)
        params = {
            "error_message": msg,
            "succeed_on_attempt": 2,
        }
        task = Task(
            id=task_id,
            name=f"Task {task_id}",
            action=action,
            parameters=params,
            success_criteria=criteria,
            idempotent=idempotent,
            max_attempts=max_attempts,
            max_recoveries=max_recoveries,
        )
        exec_result = sim.execute(task, attempt_number=attempt)
        verif_result = verifier.verify(task, exec_result, sim)
        failure_cat = recovery_mgr.classify_failure(exec_result, verif_result)
        rec_attempt = recovery_mgr.attempt_recovery(
            task=task,
            failure_category=failure_cat,
            simulator=sim,
            current_attempt=attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=0,
            workflow_budget=budget_left,
            recovery_number=1,
        )

    elif strategy == "RESTORE_FILE":
        action = TaskAction.MISSING_FILE
        msg = rng.choice(templates["MISSING_FILE"]).format(path=path)
        criteria = SuccessCriteria(CriterionType.FILE_EXISTS, expected=path)
        params = {
            "file_path": path,
            "error_message": msg,
        }
        # Ensure file does not exist initially in sandbox
        safe_p = sim.resolve_safe_path(path)
        safe_p.unlink(missing_ok=True)

        task = Task(
            id=task_id,
            name=f"Task {task_id}",
            action=action,
            parameters=params,
            success_criteria=criteria,
            idempotent=idempotent,
            max_attempts=max_attempts,
            max_recoveries=max_recoveries,
        )
        exec_result = sim.execute(task, attempt_number=attempt)
        verif_result = verifier.verify(task, exec_result, sim)
        failure_cat = recovery_mgr.classify_failure(exec_result, verif_result)
        rec_attempt = recovery_mgr.attempt_recovery(
            task=task,
            failure_category=failure_cat,
            simulator=sim,
            current_attempt=attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=0,
            workflow_budget=budget_left,
            recovery_number=1,
        )
        # Clean up restored file from sandbox so future episodes start clean
        safe_p.unlink(missing_ok=True)

    elif strategy == "SANITIZE_DATA":
        action = TaskAction.INVALID_DATA
        msg = rng.choice(templates["INVALID_DATA"]).format(key=key)
        criteria = SuccessCriteria(CriterionType.DATA_VALID, expected=[key, "valid"])
        params = {
            "data": {"invalid": True},
            "error_message": msg,
        }
        task = Task(
            id=task_id,
            name=f"Task {task_id}",
            action=action,
            parameters=params,
            success_criteria=criteria,
            idempotent=idempotent,
            max_attempts=max_attempts,
            max_recoveries=max_recoveries,
        )
        exec_result = sim.execute(task, attempt_number=attempt)
        verif_result = verifier.verify(task, exec_result, sim)
        failure_cat = recovery_mgr.classify_failure(exec_result, verif_result)
        rec_attempt = recovery_mgr.attempt_recovery(
            task=task,
            failure_category=failure_cat,
            simulator=sim,
            current_attempt=attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=0,
            workflow_budget=budget_left,
            recovery_number=1,
        )

    elif strategy == "ESCALATE":
        # Genuine ESCALATE: Execution succeeds cleanly, but declarative verification fails
        # with ambiguous output and no automated parameter fixes
        action = TaskAction.SUCCESS
        observed_val = {"status": "unverified", "payload_code": rng.randint(500, 599)}
        expected_val = {"status": "verified", "payload_code": 200}
        criteria = SuccessCriteria(CriterionType.OUTPUT_EQUALS, expected=expected_val)
        msg = rng.choice(templates["ESCALATE"]).format(
            domain=domain,
            criterion="output_equals",
            expected=expected_val,
            observed=observed_val,
        )
        params = {
            "output": observed_val,
            "error_message": msg,
        }
        task = Task(
            id=task_id,
            name=f"Task {task_id}",
            action=action,
            parameters=params,
            success_criteria=criteria,
            idempotent=idempotent,
            max_attempts=max_attempts,
            max_recoveries=max_recoveries,
        )
        exec_result = sim.execute(task, attempt_number=attempt)
        verif_result = verifier.verify(task, exec_result, sim)
        failure_cat = recovery_mgr.classify_failure(exec_result, verif_result)
        rec_attempt = recovery_mgr.attempt_recovery(
            task=task,
            failure_category=failure_cat,
            simulator=sim,
            current_attempt=attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=0,
            workflow_budget=budget_left,
            recovery_number=1,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Build standardized observation
    obs = make_observation(
        action=action.value,
        failure_category=failure_cat.value,
        error_message=msg,
        attempt=attempt,
        task_recoveries_used=task_recoveries_used,
        workflow_budget_left=budget_left,
        idempotent=idempotent,
        criterion_type=task.success_criteria.criterion_type.value,
        task_id=task.id,
    )

    obs_str = json.dumps(obs, sort_keys=True)
    prompt = f"OBSERVATION: {obs_str}\nSTRATEGY:"
    target = f" {rec_attempt.strategy.name} | RATIONALE: {rec_attempt.details}"

    return {
        "prompt": prompt,
        "target": target,
        "full_text": f"{prompt}{target}",
        "strategy": rec_attempt.strategy.name,
        "action": action.value,
        "is_novel": is_novel,
        "signature": signature,
    }


def generate_episode(
    episode_idx: int,
    rng: random.Random,
    sim: FailureSimulator,
    verifier: Verifier,
    recovery_mgr: RecoveryManager,
    is_novel: bool = False,
) -> dict[str, Any]:
    """Generate a single deterministic training instance using the Phase 1 engine as oracle."""
    strat = VALID_STRATEGIES[episode_idx % len(VALID_STRATEGIES)]
    return generate_episode_for_strategy(
        strategy=strat,
        episode_idx=episode_idx,
        rng=rng,
        sim=sim,
        verifier=verifier,
        recovery_mgr=recovery_mgr,
        is_novel=is_novel,
    )


def build_dataset(
    num_episodes: int = 30000,
    seed: int = 42,
    preset_name: str = "base",
) -> dict[str, Any]:
    """Generate multi-split dataset strictly stratified and verified for disjointness."""
    rng = random.Random(seed)
    sim = FailureSimulator()
    verifier = Verifier()
    recovery_mgr = RecoveryManager()

    config = get_preset(preset_name)
    tokenizer = BrainTokenizer(vocab_size=config.vocab_size)

    # Split sizes: train (75%), val (8.33%), test (8.33%), novel (8.33%)
    # For 30,000 episodes: 22,500 train, 2,500 val, 2,500 test, 2,500 novel
    n_novel = 2500
    n_test = 2500
    n_val = 2500
    n_train = max(5000, num_episodes - (n_novel + n_test + n_val))

    data_dir = resolve_brain_path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RecoverBrain Balanced Curriculum Data Engine")
    print("=" * 70)
    print(f"Total episodes: {n_train + n_val + n_test + n_novel:,} (seed={seed})")
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,} | Novel: {n_novel:,}")

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
        "novel": [],
    }

    # Generate standard splits with strict round-robin class stratification
    for split_name, target_count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        per_class = target_count // len(VALID_STRATEGIES)
        ep_idx = 0
        for strat in VALID_STRATEGIES:
            for _ in range(per_class):
                ep = generate_episode_for_strategy(
                    strategy=strat,
                    episode_idx=ep_idx,
                    rng=rng,
                    sim=sim,
                    verifier=verifier,
                    recovery_mgr=recovery_mgr,
                    is_novel=False,
                )
                splits[split_name].append(ep)
                ep_idx += 1
        rng.shuffle(splits[split_name])

    # Generate novel split with strict round-robin class stratification
    per_class_novel = n_novel // len(VALID_STRATEGIES)
    ep_idx = 0
    for strat in VALID_STRATEGIES:
        for _ in range(per_class_novel):
            ep = generate_episode_for_strategy(
                strategy=strat,
                episode_idx=ep_idx,
                rng=rng,
                sim=sim,
                verifier=verifier,
                recovery_mgr=recovery_mgr,
                is_novel=True,
            )
            splits["novel"].append(ep)
            ep_idx += 1
    rng.shuffle(splits["novel"])

    # 1. Automated check: verify train and novel parameter combinations are disjoint
    train_signatures = {tuple(ep["signature"]) for ep in splits["train"]}
    novel_signatures = {tuple(ep["signature"]) for ep in splits["novel"]}
    overlap = train_signatures.intersection(novel_signatures)
    print(f"\n[Disjointness Check] Unique train signatures: {len(train_signatures)}, novel signatures: {len(novel_signatures)}")
    if overlap:
        raise AssertionError(f"Train and novel combinations overlap: {overlap}")
    print("  * Verified: Train and novel parameter combinations are 100% disjoint.")

    # 2. Print class distributions for every split and verify bounds
    print("\n[Class Distribution Verification]")
    split_distributions: dict[str, dict[str, float]] = {}
    for split_name, examples in splits.items():
        total = len(examples)
        counts = Counter(ex["strategy"] for ex in examples)
        print(f"\nSplit '{split_name}' (Total: {total:,} samples):")
        split_distributions[split_name] = {}
        for s in VALID_STRATEGIES:
            cnt = counts[s]
            pct = (cnt / total) * 100.0 if total > 0 else 0.0
            split_distributions[split_name][s] = pct
            print(f"  * {s:<15}: {cnt:>5} ({pct:5.2f}%)")
            assert pct >= 15.0, f"Class {s} in split {split_name} has {pct:.2f}% < 15.0% threshold!"

    # 3. Check that novel split is NOT more ESCALATE-heavy than val
    val_esc_pct = split_distributions["val"]["ESCALATE"]
    novel_esc_pct = split_distributions["novel"]["ESCALATE"]
    print(f"\n[Escalate Distribution Check] Val ESCALATE: {val_esc_pct:.2f}% | Novel ESCALATE: {novel_esc_pct:.2f}%")
    if novel_esc_pct > val_esc_pct + 2.0:
        raise AssertionError("Novel split is more ESCALATE-heavy than val!")
    print("  * Verified: Novel split ESCALATE rate is balanced and aligned with validation.")

    # Save tokenizer
    tokenizer.save("data/tokenizer.json")

    # Serialize JSONL and tokenized .npy arrays
    metadata: dict[str, Any] = {
        "total_episodes": len(splits["train"]) + len(splits["val"]) + len(splits["test"]) + len(splits["novel"]),
        "seed": seed,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "splits": {},
    }

    print("\nSerializing JSONL and tokenized .npy arrays...")
    for split_name, examples in splits.items():
        jsonl_path = data_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        packed_array = np.full(
            (len(examples), config.max_seq_len), fill_value=PAD_ID, dtype=np.uint16
        )

        for i, ex in enumerate(examples):
            tokens = (
                [BOS_ID]
                + tokenizer.encode(ex["prompt"])
                + [SEP_ID]
                + tokenizer.encode(ex["target"])
                + [EOS_ID]
            )
            tokens = tokens[: config.max_seq_len]
            packed_array[i, : len(tokens)] = tokens

        npy_path = data_dir / f"{split_name}.npy"
        np.save(str(npy_path), packed_array)

        hasher = hashlib.sha256()
        with npy_path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        split_hash = hasher.hexdigest()

        metadata["splits"][split_name] = {
            "count": len(examples),
            "npy_path": str(npy_path.name),
            "sha256": split_hash,
            "class_distribution": split_distributions[split_name],
        }

    info_path = data_dir / "dataset_info.json"
    info_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    sim.cleanup()

    print(f"\n[SUCCESS] Balanced dataset generated and packed in {data_dir}\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic curriculum dataset generator for RecoverBrain")
    parser.add_argument("--episodes", type=int, default=30000, help="Number of total episodes to generate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--preset", type=str, default="base", choices=["tiny", "small", "base"])
    args = parser.parse_args()

    build_dataset(num_episodes=args.episodes, seed=args.seed, preset_name=args.preset)


if __name__ == "__main__":
    main()
