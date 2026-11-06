"""Advised recovery manager integrating RecoverBrain recommendations under deterministic oversight."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.failure_simulator import FailureSimulator
from app.models import (
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    Task,
    _current_timestamp,
)
from app.recovery import RecoveryManager
from app.storage import Storage
from brain.advisor import BrainAdvisor

logger = logging.getLogger("recoverops.advised_recovery")


class AdvisedRecoveryManager(RecoveryManager):
    """Subclass of RecoveryManager enforcing Phase 1 guardrails while auditing RecoverBrain proposals.

    Safety Architecture:
      1. Phase 1 guardrails (unrecoverable, non-idempotent, max attempts, max recoveries,
         shared budget) execute FIRST and UNCHANGED.
      2. If guardrails pass, the neural model is consulted for an advisory proposal.
      3. Propose-vs-Decide Rule:
         - If the model proposes ESCALATE:
           * If escalate_confidence_threshold > 0.0 and confidence < threshold:
             the proposal is overridden and the deterministic strategy is applied.
           * Otherwise, the ESCALATE proposal is accepted as a safety signal.
         - If the model proposes an action agreeing with deterministic policy, it is accepted.
         - If the model proposes an action disagreeing with deterministic policy, it is overridden
           and the deterministic strategy is applied.
      4. Every proposal is audited in the 'brain_proposals' relational table using parameterized SQL.
    """

    def __init__(
        self,
        advisor: BrainAdvisor,
        storage: Storage | None = None,
        escalate_confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.advisor = advisor
        self.storage = storage
        self.escalate_confidence_threshold = escalate_confidence_threshold
        self._init_proposals_table()

    def _init_proposals_table(self) -> None:
        """Create the brain_proposals audit table if storage is available."""
        if self.storage is None:
            return
        conn = self.storage._get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS brain_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    proposed_strategy TEXT NOT NULL,
                    final_strategy TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    override_reason TEXT,
                    model_preset TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )

    def _record_proposal(
        self,
        task_id: str,
        observation: dict[str, Any],
        proposed: str,
        final_strategy: str,
        accepted: bool,
        override_reason: str | None,
    ) -> None:
        """Persist advisory proposal into brain_proposals table using parameterized SQL."""
        if self.storage is None:
            return
        conn = self.storage._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO brain_proposals (
                    task_id,
                    observation_json,
                    proposed_strategy,
                    final_strategy,
                    accepted,
                    override_reason,
                    model_preset,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    task_id,
                    json.dumps(observation, sort_keys=True),
                    proposed,
                    final_strategy,
                    1 if accepted else 0,
                    override_reason,
                    self.advisor.preset_name,
                    _current_timestamp(),
                ),
            )

    def attempt_recovery(
        self,
        task: Task,
        failure_category: FailureCategory,
        simulator: FailureSimulator,
        current_attempt: int,
        task_recoveries_used: int,
        workflow_recoveries_used: int,
        workflow_budget: int,
        recovery_number: int,
    ) -> RecoveryAttempt:
        """Evaluate guardrails first; consult advisor only at strategy selection."""
        # 1. Evaluate all Phase 1 guardrails FIRST and UNCHANGED
        guardrail_attempt = super().attempt_recovery(
            task=task,
            failure_category=failure_category,
            simulator=simulator,
            current_attempt=current_attempt,
            task_recoveries_used=task_recoveries_used,
            workflow_recoveries_used=workflow_recoveries_used,
            workflow_budget=workflow_budget,
            recovery_number=recovery_number,
        )

        # If guardrails triggered escalation, return immediately
        if guardrail_attempt.escalated or not guardrail_attempt.applied:
            return guardrail_attempt

        # 2. Guardrails passed: Consult RecoverBrain for proposal
        deterministic_strategy = guardrail_attempt.strategy.name
        observation = {
            "action": task.action.value,
            "attempt": current_attempt,
            "criterion_type": task.success_criteria.criterion_type.value,
            "error_message": task.parameters.get("error_message", ""),
            "failure_category": failure_category.value,
            "idempotent": task.idempotent,
            "task_id": task.id,
            "task_recoveries_used": task_recoveries_used,
            "workflow_budget_left": workflow_budget - workflow_recoveries_used,
        }

        try:
            res = self.advisor.propose_strategy(observation)
            if len(res) == 3:
                proposed_name, rationale, confidence = res
            else:
                proposed_name, rationale = res  # type: ignore
                confidence = 1.0
        except Exception as e:
            logger.warning("Advisor inference failed (%s); defaulting to deterministic strategy", e)
            proposed_name, rationale, confidence = "ESCALATE", f"Advisor inference error: {e}", 0.0

        # 3. Apply safety arbitration rule
        accepted = False
        override_reason: str | None = None
        final_strategy_name = deterministic_strategy

        if proposed_name == "ESCALATE":
            if self.escalate_confidence_threshold > 0.0 and confidence < self.escalate_confidence_threshold:
                # Confidence too low to honor ESCALATE; fall back to deterministic policy
                final_strategy_name = deterministic_strategy
                accepted = False
                override_reason = (
                    f"Model proposed ESCALATE with confidence {confidence:.3f} below threshold "
                    f"{self.escalate_confidence_threshold:.3f}; falling back to deterministic policy '{deterministic_strategy}'"
                )
            else:
                final_strategy_name = "ESCALATE"
                accepted = True
                override_reason = "Model advisory proposed safety escalation; honored by policy"
        elif proposed_name == deterministic_strategy:
            accepted = True
            override_reason = None
        else:
            # Model proposed action differing from deterministic policy
            accepted = False
            final_strategy_name = deterministic_strategy
            override_reason = (
                f"Model proposed '{proposed_name}' but deterministic policy requires '{deterministic_strategy}'"
            )

        # 4. Audit proposal
        self._record_proposal(
            task_id=task.id,
            observation=observation,
            proposed=proposed_name,
            final_strategy=final_strategy_name,
            accepted=accepted,
            override_reason=override_reason,
        )

        # 5. Dispatch selected remediation
        if final_strategy_name == "ESCALATE":
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details=f"Escalated per AI proposal. Rationale: {rationale}",
                escalated=True,
            )

        # Return original validated recovery attempt with enriched details
        details_augmented = (
            f"{guardrail_attempt.details} [AI Advisor: proposed={proposed_name}, accepted={accepted}]"
        )
        return RecoveryAttempt(
            task_id=task.id,
            recovery_number=recovery_number,
            failure_category=failure_category,
            strategy=guardrail_attempt.strategy,
            applied=guardrail_attempt.applied,
            details=details_augmented,
            escalated=False,
        )
