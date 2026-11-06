"""Synthetic curriculum data engine with sequential history-dependent oracle policy.

Stage 2 Oracle Policies:
  - Observation includes rich execution history: prior strategies tried, outcome success,
    attempt number, task recoveries used, workflow budget left, and noise/distractor fields.
  - Sequential Oracle Rules:
    1. Loop Prevention: If strategy S was already applied and failure recurred -> ESCALATE.
    2. Persistent Fault: If RESET_SERVICE was applied and SERVICE_READY failed twice -> ESCALATE.
    3. Timeout Conservation: If TIMEOUT recurs at attempt == max_attempts - 1 -> ESCALATE early.
    4. Ambiguous Disambiguation: For VERIFICATION_FAILURE:
       - SANITIZE_DATA if sanitized_data configured and unattempted;
       - RETRY if transient behavior configured;
       - else ESCALATE.
    5. Conflicting Signal Priority: Prioritized resolution over ambient noisy error text.
    6. Distractor Injection: Ambient load, region, pid, cache hit rate to require structured context.
    7. Sanitized Task IDs: Completely neutral IDs with zero strategy tokens to prevent label leakage.
  - Held-out generalization family: DISK_QUOTA (target PURGE_TEMP) never seen in training.
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

# Supported strategies in training curriculum (DISK_QUOTA / PURGE_TEMP is held out)
TRAIN_STRATEGIES = [
    "RETRY",
    "RESET_SERVICE",
    "RESTORE_FILE",
    "SANITIZE_DATA",
    "ROLLBACK_REPLAY",
    "ESCALATE",
]

ALL_STRATEGIES = TRAIN_STRATEGIES + ["PURGE_TEMP"]

STANDARD_DOMAINS = [
    "auth-gateway",
    "billing-worker",
    "inventory-api",
    "order-router",
    "customer-cache",
    "session-store",
    "notification-hub",
    "payment-processor",
    "ledger-service",
    "stream-ingest",
]

NOVEL_DOMAINS = [
    "quantum-vault",
    "satellite-telemetry",
    "biometric-auth",
    "distributed-consensus",
    "pci-firewall",
    "firmware-flasher",
    "edge-mesh",
    "orbital-link",
]

STANDARD_PORTS = [8080, 8081, 5000, 3000, 9000, 8443, 5432, 6379]
NOVEL_PORTS = [9443, 7077, 4430, 27017, 11434, 18080, 28015, 3128]

STANDARD_PATHS = [
    "configs/settings.json",
    "data/payload.json",
    "assets/manifest.json",
    "reports/export.json",
    "schemas/input.json",
    "logs/application.log",
    "cache/state.json",
]

NOVEL_PATHS = [
    "vault/secure_token.pem",
    "telemetry/stream.bin",
    "snapshots/cluster_state.parquet",
    "firmware/device.bin",
    "pci/audit_trail.log",
    "keys/hardware_key.dat",
    "enclave/cert.der",
]

STANDARD_KEYS = ["id", "valid", "checksum", "user_id", "email", "account_ref"]
NOVEL_KEYS = ["block_hash", "entropy_sig", "device_fingerprint", "nonce_token", "shard_id"]

REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1", "ca-central-1"]


def make_neutral_task_id(rng: random.Random, prefix: str = "t") -> str:
    """Generate completely neutral task identifier without any strategy token."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    suffix = "".join(rng.choice(chars) for _ in range(6))
    return f"{prefix}_{suffix}"


def generate_episode(
    target_strategy_or_idx: str | int = "RETRY",
    episode_idx_or_rng: int | random.Random | None = None,
    rng: random.Random | None = None,
    sim: FailureSimulator | None = None,
    verifier: Verifier | None = None,
    recovery_mgr: RecoveryManager | None = None,
    is_novel: bool = False,
    is_held_out: bool = False,
) -> dict[str, Any]:
    """Generate a single curriculum episode governed by the sequential oracle.

    Supports legacy positional call: generate_episode(idx, rng, sim, verifier, rm, is_novel)
    as well as Stage 2 call: generate_episode(strategy, idx, rng, is_novel=..., is_held_out=...)
    """
    if isinstance(target_strategy_or_idx, int):
        episode_idx = target_strategy_or_idx
        if isinstance(episode_idx_or_rng, random.Random):
            actual_rng = episode_idx_or_rng
        elif rng is not None:
            actual_rng = rng
        else:
            actual_rng = random.Random(episode_idx)
        target_strategy = actual_rng.choice(TRAIN_STRATEGIES)
    else:
        target_strategy = str(target_strategy_or_idx)
        if isinstance(episode_idx_or_rng, int):
            episode_idx = episode_idx_or_rng
        else:
            episode_idx = 0
        if rng is not None:
            actual_rng = rng
        elif isinstance(episode_idx_or_rng, random.Random):
            actual_rng = episode_idx_or_rng
        else:
            actual_rng = random.Random(episode_idx)

    rng = actual_rng
    domains = NOVEL_DOMAINS if is_novel else STANDARD_DOMAINS
    ports = NOVEL_PORTS if is_novel else STANDARD_PORTS
    paths = NOVEL_PATHS if is_novel else STANDARD_PATHS
    keys = NOVEL_KEYS if is_novel else STANDARD_KEYS

    domain = rng.choice(domains)
    port = rng.choice(ports)
    path = rng.choice(paths)
    key = rng.choice(keys)
    region = rng.choice(REGIONS)
    load_pct = round(rng.uniform(12.0, 94.0), 1)
    task_id = make_neutral_task_id(rng, prefix="job")

    # Guardrails baseline: all Phase 1 guardrails must pass
    idempotent = True
    max_attempts = rng.randint(4, 6)
    max_recoveries = rng.randint(3, 5)

    is_history_dependent = False
    is_ambiguous = False
    is_conflicting = False
    family = "standard"

    # Held-out family: DISK_QUOTA -> PURGE_TEMP
    if is_held_out or target_strategy == "PURGE_TEMP":
        family = "disk_quota"
        quota_errs = [
            f"Disk quota exceeded on volume /var/sandbox/{domain}: storage capacity reached 99.8%",
            f"IOError: No space left on device while staging artifacts for {domain}",
            f"Storage quota exhausted: temporary disk buffer full in {region}",
            f"Volume write failure: sandbox workspace allocated blocks depleted on {domain}",
        ]
        obs = {
            "action": "disk_quota",
            "attempt": 1,
            "criterion_type": "output_equals",
            "error_message": rng.choice(quota_errs),
            "failure_category": "disk_quota_exceeded",
            "history": [],
            "idempotent": True,
            "region": region,
            "ambient_load_pct": load_pct,
            "task_id": task_id,
            "task_recoveries_used": 0,
            "workflow_budget_left": rng.randint(2, 5),
        }
        target = "PURGE_TEMP"
        rationale = f"Purge sandbox-registered temporary artifacts to relieve volume exhaustion on {domain}"
        prompt = f"OBSERVATION: {json.dumps(obs, sort_keys=True)}\nSTRATEGY:"
        full_text = f"{prompt}{target} | RATIONALE: {rationale}"
        return {
            "prompt": prompt,
            "target": f"{target} | RATIONALE: {rationale}",
            "full_text": full_text,
            "strategy": target,
            "observation": obs,
            "rationale": rationale,
            "is_novel": is_novel,
            "is_history_dependent": False,
            "is_ambiguous": False,
            "is_conflicting": False,
            "family": family,
        }

    # Standard training strategies
    if target_strategy == "RETRY":
        sub_type = rng.choice(["timeout", "rate_limit", "ambiguous_transient"])
        if sub_type == "timeout":
            family = "timeout"
            attempt = 1
            rec_used = 0
            err_msg = rng.choice([
                f"Operation on {domain}:{port} timed out after 3000ms",
                f"Deadline exceeded on remote RPC call to {domain}",
                f"Upstream socket idle timeout breached SLA on {domain}",
            ])
            obs = {
                "action": "timeout",
                "attempt": attempt,
                "criterion_type": "output_contains",
                "error_message": err_msg,
                "failure_category": "timeout_error",
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": rec_used,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Retry idempotent operation for timeout on {domain}"
        elif sub_type == "rate_limit":
            family = "rate_limited"
            attempt = 1
            rec_used = 0
            err_msg = rng.choice([
                f"Rate limit exceeded (HTTP 429) on API gateway {domain}:{port}",
                f"Too many requests to {domain}: token bucket depleted, backoff step 1",
                f"Throttling triggered on {domain}: concurrency limit breached",
            ])
            obs = {
                "action": "rate_limited",
                "attempt": attempt,
                "criterion_type": "output_contains",
                "error_message": err_msg,
                "failure_category": "rate_limit_exceeded",
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": rec_used,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Retry with logical backoff counter for rate-limited endpoint {domain}"
        else:
            # Ambiguous verification with transient flag (Rule 4)
            family = "ambiguous_transient"
            is_ambiguous = True
            attempt = 1
            rec_used = 0
            err_msg = f"Output verification mismatch on {domain}: expected valid state but observed transient null"
            obs = {
                "action": "transient_service",
                "attempt": attempt,
                "criterion_type": "output_equals",
                "error_message": err_msg,
                "failure_category": "verification_failure",
                "transient_configured": True,
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": rec_used,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Retry transient verification failure where downstream recovery is configured for {domain}"

    elif target_strategy == "RESET_SERVICE":
        family = "transient_service"
        attempt = 1
        rec_used = 0
        err_msg = rng.choice([
            f"Connection refused by {domain} on port {port}: socket closed",
            f"Service {domain} temporarily unavailable: initializing thread pool",
            f"Health probe on {domain}:{port} returned 503 Service Unavailable",
            f"Internal cluster transport error: connection dropped by peer {domain}",
        ])
        # Sometimes inject conflicting wording (Rule 5)
        if rng.random() < 0.25:
            is_conflicting = True
            err_msg += f" (Note: previous operation reported read timeout)"

        obs = {
            "action": "transient_service",
            "attempt": attempt,
            "criterion_type": "service_ready",
            "error_message": err_msg,
            "failure_category": "transient_service_error",
            "history": [],
            "idempotent": True,
            "region": region,
            "ambient_load_pct": load_pct,
            "task_id": task_id,
            "task_recoveries_used": rec_used,
            "workflow_budget_left": rng.randint(2, 4),
        }
        rationale = f"Reset service runtime state to restore health for {domain}"

    elif target_strategy == "RESTORE_FILE":
        family = "missing_file"
        attempt = 1
        rec_used = 0
        err_msg = rng.choice([
            f"Required asset file missing: {path}",
            f"Storage dependency {path} not found in container filesystem",
            f"Critical configuration {path} is absent from local path",
            f"Path lookup failed: asset '{path}' does not exist on {domain}",
        ])
        obs = {
            "action": "missing_file",
            "attempt": attempt,
            "criterion_type": "file_exists",
            "error_message": err_msg,
            "failure_category": "missing_file_error",
            "history": [],
            "idempotent": True,
            "region": region,
            "ambient_load_pct": load_pct,
            "task_id": task_id,
            "task_recoveries_used": rec_used,
            "workflow_budget_left": rng.randint(2, 4),
        }
        rationale = f"Restore missing file '{path}' from backup snapshot"

    elif target_strategy == "SANITIZE_DATA":
        sub_type = rng.choice(["direct_invalid", "ambiguous_sanitized"])
        if sub_type == "direct_invalid":
            family = "invalid_data"
            attempt = 1
            rec_used = 0
            err_msg = rng.choice([
                f"Data validation failed: missing key '{key}' in record",
                f"Schema corruption: null value found for required field '{key}'",
                f"Structural contract violation on field '{key}' from {domain}",
                f"Checksum mismatch on ingested record for key '{key}'",
            ])
            obs = {
                "action": "invalid_data",
                "attempt": attempt,
                "criterion_type": "data_valid",
                "error_message": err_msg,
                "failure_category": "invalid_data_error",
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": rec_used,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Sanitize malformed payload schema for key '{key}'"
        else:
            # Ambiguous verification failure with sanitized data configured (Rule 4)
            family = "ambiguous_verification"
            is_ambiguous = True
            attempt = 1
            rec_used = 0
            err_msg = f"Output verification mismatch on {domain}: observed payload missing clean keys"
            obs = {
                "action": "invalid_data",
                "attempt": attempt,
                "criterion_type": "data_valid",
                "error_message": err_msg,
                "failure_category": "verification_failure",
                "sanitized_data_configured": True,
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": rec_used,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Sanitize payload according to configured template for task '{task_id}'"

    elif target_strategy == "ROLLBACK_REPLAY":
        family = "deadlock"
        attempt = 1
        rec_used = 0
        err_msg = rng.choice([
            f"Transaction deadlock detected on table lock in SQLite sandbox on {domain}",
            f"OperationalError: database table locked by concurrent process in {region}",
            f"Deadlock mutual exclusion detected during transaction commit on {domain}",
            f"Write conflict: transactional lock acquisition timed out on SQLite engine",
        ])
        obs = {
            "action": "deadlock",
            "attempt": attempt,
            "criterion_type": "output_equals",
            "error_message": err_msg,
            "failure_category": "transaction_deadlock",
            "history": [],
            "idempotent": True,
            "region": region,
            "ambient_load_pct": load_pct,
            "task_id": task_id,
            "task_recoveries_used": rec_used,
            "workflow_budget_left": rng.randint(2, 4),
        }
        rationale = f"Rollback conflicting transaction and replay atomically on {domain}"

    elif target_strategy == "ESCALATE":
        # Oracle escalation cases: History loops, Persistent reset failure, Timeout early escalation, or Unresolvable ambiguity
        escalation_rule = rng.choice(["rule1_loop", "rule2_reset_twice", "rule3_timeout_near_limit", "rule4_ambiguous_escalate"])
        is_history_dependent = True

        if escalation_rule == "rule1_loop":
            family = "history_loop"
            prior_strat = rng.choice(["RETRY", "RESTORE_FILE", "SANITIZE_DATA", "ROLLBACK_REPLAY"])
            err_msg = f"Recurrent failure on {domain}: same error recurred after {prior_strat}"
            obs = {
                "action": "timeout" if prior_strat == "RETRY" else "missing_file",
                "attempt": 2,
                "criterion_type": "output_equals",
                "error_message": err_msg,
                "failure_category": "timeout_error" if prior_strat == "RETRY" else "missing_file_error",
                "history": [{"attempt": 1, "strategy": prior_strat, "succeeded": False}],
                "same_failure_recurred": True,
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": 1,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Repeated failure recurred after strategy '{prior_strat}'; escalating to prevent ineffective looping"

        elif escalation_rule == "rule2_reset_twice":
            family = "persistent_service_fault"
            err_msg = f"Service probe on {domain}:{port} still failing 503 after reset remediation"
            obs = {
                "action": "transient_service",
                "attempt": 3,
                "criterion_type": "service_ready",
                "error_message": err_msg,
                "failure_category": "transient_service_error",
                "history": [
                    {"attempt": 1, "strategy": "RESET_SERVICE", "succeeded": False},
                    {"attempt": 2, "strategy": "RESET_SERVICE", "succeeded": False},
                ],
                "same_failure_recurred": True,
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": 2,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Service on {domain} persistently failing after 2 reset attempts; human escalation required"

        elif escalation_rule == "rule3_timeout_near_limit":
            family = "timeout_conservation"
            # attempt == max_attempts - 1 (e.g. attempt 4 with max_attempts 5)
            attempt = max_attempts - 1
            err_msg = f"Persistent timeout on {domain}:{port}; attempt {attempt} of {max_attempts}"
            obs = {
                "action": "timeout",
                "attempt": attempt,
                "criterion_type": "output_contains",
                "error_message": err_msg,
                "failure_category": "timeout_error",
                "history": [{"attempt": i, "strategy": "RETRY", "succeeded": False} for i in range(1, attempt)],
                "same_failure_recurred": True,
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": attempt - 1,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Recurring timeout reached attempt limit minus one ({attempt}/{max_attempts}); escalating early to conserve last attempt"

        else:
            family = "unresolvable_ambiguity"
            is_ambiguous = True
            is_history_dependent = False
            err_msg = f"Output verification mismatch on {domain}: unresolvable semantic discrepancy observed"
            obs = {
                "action": "success",
                "attempt": 1,
                "criterion_type": "output_equals",
                "error_message": err_msg,
                "failure_category": "verification_failure",
                "sanitized_data_configured": False,
                "transient_configured": False,
                "history": [],
                "idempotent": True,
                "region": region,
                "ambient_load_pct": load_pct,
                "task_id": task_id,
                "task_recoveries_used": 0,
                "workflow_budget_left": rng.randint(2, 4),
            }
            rationale = f"Verification failure has no automated remediation configured for task '{task_id}'; escalating to human"

    else:
        raise ValueError(f"Unknown target strategy: {target_strategy}")

    target = target_strategy
    prompt = f"OBSERVATION: {json.dumps(obs, sort_keys=True)}\nSTRATEGY:"
    full_text = f"{prompt}{target} | RATIONALE: {rationale}"

    return {
        "prompt": prompt,
        "target": f"{target} | RATIONALE: {rationale}",
        "full_text": full_text,
        "strategy": target,
        "observation": obs,
        "rationale": rationale,
        "is_novel": is_novel,
        "is_history_dependent": is_history_dependent,
        "is_ambiguous": is_ambiguous,
        "is_conflicting": is_conflicting,
        "family": family,
    }


def generate_curriculum_dataset(
    train_count: int = 6000,
    val_count: int = 1200,
    test_count: int = 1200,
    novel_count: int = 1200,
    held_out_count: int = 600,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate balanced multi-family dataset with sequential history dependencies and held-out generalization."""
    if output_dir is None:
        out_path = resolve_brain_path("data")
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    n_train_strats = len(TRAIN_STRATEGIES)

    assert train_count % n_train_strats == 0, f"train_count must divide by {n_train_strats}"
    assert val_count % n_train_strats == 0, f"val_count must divide by {n_train_strats}"
    assert test_count % n_train_strats == 0, f"test_count must divide by {n_train_strats}"
    assert novel_count % n_train_strats == 0, f"novel_count must divide by {n_train_strats}"

    def build_split(count: int, is_novel: bool = False) -> list[dict[str, Any]]:
        per_class = count // n_train_strats
        episodes = []
        for strat in TRAIN_STRATEGIES:
            for i in range(per_class):
                ep = generate_episode(strat, episode_idx=len(episodes), rng=rng, is_novel=is_novel)
                episodes.append(ep)
        rng.shuffle(episodes)
        return episodes

    print(f"[Datagen Stage 2] Generating balanced splits across {n_train_strats} training strategies...")
    train_eps = build_split(train_count, is_novel=False)
    val_eps = build_split(val_count, is_novel=False)
    test_eps = build_split(test_count, is_novel=False)
    novel_eps = build_split(novel_count, is_novel=True)

    # Generate held-out DISK_QUOTA dataset (target = PURGE_TEMP)
    print(f"[Datagen Stage 2] Generating {held_out_count} held-out DISK_QUOTA episodes (never in train)...")
    held_out_eps = [
        generate_episode("PURGE_TEMP", episode_idx=i, rng=rng, is_novel=False, is_held_out=True)
        for i in range(held_out_count)
    ]
    rng.shuffle(held_out_eps)

    # Build and save tokenizer
    print("[Datagen Stage 2] Building vocabulary and BrainTokenizer...")
    tokenizer = BrainTokenizer(vocab_size=4096)
    tokenizer.save(out_path / "tokenizer.json")

    # Save splits as JSONL and tokenized .npy
    splits = {
        "train": train_eps,
        "val": val_eps,
        "test": test_eps,
        "novel": novel_eps,
        "held_out_disk_quota": held_out_eps,
    }

    distribution_report = {}

    for name, eps in splits.items():
        jsonl_path = out_path / f"{name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep) + "\n")

        # Distribution check
        strat_counts = Counter(e["strategy"] for e in eps)
        dist = {k: f"{v}/{len(eps)} ({v/len(eps)*100:.1f}%)" for k, v in sorted(strat_counts.items())}
        distribution_report[name] = dist
        print(f"  Split '{name}': {len(eps)} episodes | Distribution: {dist}")

        # Tokenize and save .npy
        tokenized_seqs = []
        max_seq_len = 256
        for ep in eps:
            prompt_toks = [BOS_ID] + tokenizer.encode(ep["prompt"])[:180] + [SEP_ID]
            target_toks = tokenizer.encode(ep["target"])[: (max_seq_len - len(prompt_toks) - 1)] + [EOS_ID]
            seq = prompt_toks + target_toks
            if len(seq) < max_seq_len:
                seq += [PAD_ID] * (max_seq_len - len(seq))
            tokenized_seqs.append(seq[:max_seq_len])

        npy_path = out_path / f"{name}.npy"
        np.save(str(npy_path), np.array(tokenized_seqs, dtype=np.int32))

    info = {
        "generated_at": rng.randint(1000, 9999),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "novel_count": novel_count,
        "held_out_count": held_out_count,
        "train_strategies": TRAIN_STRATEGIES,
        "held_out_strategy": "PURGE_TEMP",
        "distributions": distribution_report,
    }
    with (out_path / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


if __name__ == "__main__":
    generate_curriculum_dataset()
