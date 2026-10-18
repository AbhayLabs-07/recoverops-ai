"""Failure classification and bounded, predefined recovery strategies."""

from __future__ import annotations

from dataclasses import dataclass

from .failure_simulator import FailureSimulator
from .models import (
    FailureKind,
    RecoveryAttempt,
    RecoveryStrategy,
    Task,
    TaskAction,
    utc_now,
)


@dataclass(frozen=True)
class RecoveryDecision:
    """A classified failure and the safe strategy selected for it."""

    strategy: RecoveryStrategy
    reason: str


class RecoveryManager:
    """Choose and apply only known, bounded recovery operations."""

    def decide(self, task: Task, failure_kind: FailureKind) -> RecoveryDecision:
        """Classify a failure without executing anything."""

        if not task.idempotent:
            return RecoveryDecision(
                RecoveryStrategy.ESCALATE,
                "non-idempotent task will not be retried blindly",
            )
        if failure_kind is FailureKind.TRANSIENT_SERVICE:
            return RecoveryDecision(
                RecoveryStrategy.RESET_SERVICE,
                "reset the known service state before retrying",
            )
        if failure_kind is FailureKind.TIMEOUT:
            return RecoveryDecision(
                RecoveryStrategy.RETRY,
                "retry the known timeout without repeating arbitrary side effects",
            )
        if failure_kind is FailureKind.MISSING_FILE:
            return RecoveryDecision(
                RecoveryStrategy.RESTORE_FILE,
                "restore the explicitly configured file target",
            )
        if failure_kind is FailureKind.INVALID_DATA:
            return RecoveryDecision(
                RecoveryStrategy.SANITIZE_DATA,
                "sanitize the known data payload before retrying",
            )
        if failure_kind is FailureKind.VERIFICATION_FAILED:
            if task.action is TaskAction.INVALID_DATA:
                return RecoveryDecision(
                    RecoveryStrategy.SANITIZE_DATA,
                    "verification rejected invalid data; sanitize the known payload",
                )
            return RecoveryDecision(
                RecoveryStrategy.RETRY,
                "verification failed; retry the idempotent known action",
            )
        return RecoveryDecision(
            RecoveryStrategy.ESCALATE,
            "failure is not in the safe automatic recovery catalog",
        )

    def apply(
        self,
        task: Task,
        workflow_id: str,
        attempt_number: int,
        failure_kind: FailureKind,
        simulator: FailureSimulator,
    ) -> RecoveryAttempt:
        """Apply a selected strategy and return an auditable attempt."""

        decision = self.decide(task, failure_kind)
        if decision.strategy is RecoveryStrategy.ESCALATE:
            return RecoveryAttempt(
                workflow_id=workflow_id,
                task_id=task.id,
                attempt=attempt_number,
                strategy=decision.strategy,
                reason=decision.reason,
                succeeded=False,
                message="human intervention required",
                timestamp=utc_now(),
            )

        succeeded, message = simulator.apply_recovery(task, decision.strategy, workflow_id)
        return RecoveryAttempt(
            workflow_id=workflow_id,
            task_id=task.id,
            attempt=attempt_number,
            strategy=decision.strategy,
            reason=decision.reason,
            succeeded=succeeded,
            message=message,
            timestamp=utc_now(),
        )
