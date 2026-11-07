"""Outcome-derived rollout oracle evaluating candidate recovery strategies via simulated lookahead.

For each decision state, RolloutOracle:
  1. Clones the deterministic simulator state into an isolated sandbox replica.
  2. Rolls out each candidate strategy: RETRY, RESET_SERVICE, RESTORE_FILE,
     SANITIZE_DATA, ROLLBACK_REPLAY, PURGE_TEMP, ESCALATE through real Part A components.
  3. Executes the subsequent attempt and evaluates declarative verification criteria.
  4. Scores each candidate by:
       - Verified Success (+100)
       - Cost penalty based on attempts and recoveries consumed
       - Safety (guardrail violations or sandbox escapes penalized with -inf)
  5. Determines the optimal label with a deterministic, documented tie-break.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import shutil
from typing import Any

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    VerificationResult,
)
from app.recovery import RecoveryManager
from app.verification import Verifier


TIE_BREAK_PRIORITY = [
    RecoveryStrategy.RETRY,
    RecoveryStrategy.RESET_SERVICE,
    RecoveryStrategy.RESTORE_FILE,
    RecoveryStrategy.SANITIZE_DATA,
    RecoveryStrategy.ROLLBACK_REPLAY,
    RecoveryStrategy.PURGE_TEMP,
    RecoveryStrategy.ESCALATE,
]


@dataclass(frozen=True)
class RolloutScore:
    """Evaluation score for a candidate recovery strategy."""
    strategy: RecoveryStrategy
    score: float
    verified: bool
    safe: bool
    details: str


class RolloutOracle:
    """Deterministically evaluates strategies by outcome-derived simulation rollouts."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="rollout_oracle_")).resolve() if root_dir is None else Path(root_dir).resolve()
        self.verifier = Verifier()
        self.recovery_mgr = RecoveryManager()

    def cleanup(self) -> None:
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def _clone_simulator(self, original_sim: FailureSimulator) -> FailureSimulator:
        """Create an isolated cloned simulator with identical state dictionaries."""
        clone_dir = self.temp_root / f"clone_{id(original_sim)}_{len(original_sim._service_ready)}_{len(original_sim._logical_backoffs)}"
        clone_dir.mkdir(parents=True, exist_ok=True)
        clone = FailureSimulator(root_dir=clone_dir)

        # Copy state maps
        clone._service_ready = dict(original_sim._service_ready)
        clone._sanitized_data = {k: copy.deepcopy(v) for k, v in original_sim._sanitized_data.items()}
        clone._restored_files = set(original_sim._restored_files)
        clone._logical_backoffs = dict(original_sim._logical_backoffs)
        clone._deadlocks_resolved = set(original_sim._deadlocks_resolved)
        clone._sandbox_temp_files = set(original_sim._sandbox_temp_files)
        clone._quota_resolved = set(original_sim._quota_resolved)

        # Copy physical registered files if any
        if original_sim.root_dir.exists():
            for f in original_sim.root_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(original_sim.root_dir)
                    dest = clone.root_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.read_bytes())

        return clone

    def evaluate_strategy(
        self,
        strategy: RecoveryStrategy,
        task: Task,
        failure_category: FailureCategory,
        simulator: FailureSimulator,
        current_attempt: int,
        task_recoveries_used: int,
        workflow_recoveries_used: int,
        workflow_budget: int,
    ) -> RolloutScore:
        """Roll out a single strategy in an isolated cloned sandbox and return its objective score."""
        # Safety Guardrails
        if failure_category == FailureCategory.UNRECOVERABLE_ERROR and strategy != RecoveryStrategy.ESCALATE:
            return RolloutScore(strategy, score=-float("inf"), verified=False, safe=False, details="Violates unrecoverable guardrail")

        if not task.idempotent and strategy != RecoveryStrategy.ESCALATE:
            return RolloutScore(strategy, score=-float("inf"), verified=False, safe=False, details="Violates non-idempotent guardrail")

        if strategy != RecoveryStrategy.ESCALATE:
            if current_attempt >= task.max_attempts:
                return RolloutScore(strategy, score=-float("inf"), verified=False, safe=False, details="Violates max_attempts guardrail")
            if task_recoveries_used >= task.max_recoveries:
                return RolloutScore(strategy, score=-float("inf"), verified=False, safe=False, details="Violates max_recoveries guardrail")
            if workflow_recoveries_used >= workflow_budget:
                return RolloutScore(strategy, score=-float("inf"), verified=False, safe=False, details="Violates workflow budget guardrail")

        if strategy == RecoveryStrategy.ESCALATE:
            # Baseline safe abstention score: 50.0
            return RolloutScore(strategy, score=50.0, verified=False, safe=True, details="Safe escalation to human operator")

        # Roll out recovery strategy on cloned simulator
        clone_sim = self._clone_simulator(simulator)
        try:
            if strategy == RecoveryStrategy.RETRY:
                pass  # Idempotent retry has no mutator
            elif strategy == RecoveryStrategy.RESET_SERVICE:
                clone_sim.reset_service(task)
            elif strategy == RecoveryStrategy.RESTORE_FILE:
                clone_sim.restore_file(task)
            elif strategy == RecoveryStrategy.SANITIZE_DATA:
                clone_sim.sanitize_data(task)
            elif strategy == RecoveryStrategy.ROLLBACK_REPLAY:
                clone_sim.rollback_replay(task)
            elif strategy == RecoveryStrategy.PURGE_TEMP:
                clone_sim.purge_temp(task)
        except Exception as e:
            clone_sim.cleanup()
            return RolloutScore(strategy, score=10.0, verified=False, safe=True, details=f"Remediation raised error: {e}")

        # Execute subsequent attempt
        next_attempt = current_attempt + 1
        exec_res = clone_sim.execute(task, attempt_number=next_attempt)

        # Verify outcome
        if exec_res.exception_raised:
            clone_sim.cleanup()
            return RolloutScore(strategy, score=20.0, verified=False, safe=True, details=f"Subsequent attempt failed: {exec_res.error_message}")

        v_res = self.verifier.verify(task, exec_res, clone_sim)
        clone_sim.cleanup()

        if v_res.passed:
            # Verified success: +100 minus cost of attempts and recoveries
            cost = (next_attempt * 2.0) + ((task_recoveries_used + 1) * 3.0)
            score = 100.0 - cost
            return RolloutScore(strategy, score=score, verified=True, safe=True, details=f"Verified success on attempt {next_attempt}")
        else:
            return RolloutScore(strategy, score=25.0, verified=False, safe=True, details=f"Verification failed: {v_res.details}")

    def select_best_strategy(
        self,
        task: Task,
        failure_category: FailureCategory,
        simulator: FailureSimulator,
        current_attempt: int,
        task_recoveries_used: int,
        workflow_recoveries_used: int,
        workflow_budget: int,
        candidate_strategies: list[RecoveryStrategy] | None = None,
    ) -> tuple[RecoveryStrategy, list[RolloutScore]]:
        """Evaluate all allowed candidate strategies and select the best according to outcome score and tie-break."""
        if candidate_strategies is None:
            candidate_strategies = list(TIE_BREAK_PRIORITY)

        scores = []
        for strat in candidate_strategies:
            sc = self.evaluate_strategy(
                strategy=strat,
                task=task,
                failure_category=failure_category,
                simulator=simulator,
                current_attempt=current_attempt,
                task_recoveries_used=task_recoveries_used,
                workflow_recoveries_used=workflow_recoveries_used,
                workflow_budget=workflow_budget,
            )
            scores.append(sc)

        # Find maximum score
        max_score = max(s.score for s in scores)
        top_candidates = [s.strategy for s in scores if abs(s.score - max_score) < 1e-6]

        # Apply deterministic tie-break priority
        best_strat = RecoveryStrategy.ESCALATE
        for p in TIE_BREAK_PRIORITY:
            if p in top_candidates:
                best_strat = p
                break

        return best_strat, scores

