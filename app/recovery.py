"""Deterministic recovery strategy evaluation, guardrails, and remediation dispatch."""

from __future__ import annotations

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    Task,
    VerificationResult,
)


class RecoveryManager:
    """Manages failure classification and bounded automated recovery.

    Guarantees:
      - Recovery guardrails execute in strict precedence order.
      - Budget counters are checked by RecoveryManager, but mutated ONLY by WorkflowExecutor.
      - Non-idempotent operations or unrecoverable conditions are escalated immediately.
    """

    def classify_failure(
        self,
        execution: ExecutionResult,
        verification: VerificationResult | None,
    ) -> FailureCategory:
        """Classify a failure into a domain category based on execution and verification results."""
        if execution.exception_raised:
            return execution.error_code

        if verification is not None and not verification.passed:
            crit = verification.criterion_type
            if crit == CriterionType.DATA_VALID:
                return FailureCategory.INVALID_DATA_ERROR
            if crit == CriterionType.FILE_EXISTS:
                return FailureCategory.MISSING_FILE_ERROR
            if crit == CriterionType.SERVICE_READY:
                return FailureCategory.TRANSIENT_SERVICE_ERROR
            return FailureCategory.VERIFICATION_FAILURE

        return FailureCategory.NONE

    def determine_strategy(
        self,
        task: Task,
        failure_category: FailureCategory,
    ) -> RecoveryStrategy:
        """Map a failure category to an appropriate automated recovery strategy."""
        if failure_category == FailureCategory.TIMEOUT_ERROR:
            return RecoveryStrategy.RETRY

        if failure_category == FailureCategory.TRANSIENT_SERVICE_ERROR:
            return RecoveryStrategy.RESET_SERVICE

        if failure_category == FailureCategory.MISSING_FILE_ERROR:
            return RecoveryStrategy.RESTORE_FILE

        if failure_category == FailureCategory.INVALID_DATA_ERROR:
            return RecoveryStrategy.SANITIZE_DATA

        if failure_category == FailureCategory.UNRECOVERABLE_ERROR:
            return RecoveryStrategy.ESCALATE

        if failure_category == FailureCategory.RATE_LIMIT_EXCEEDED:
            return RecoveryStrategy.RETRY

        if failure_category == FailureCategory.TRANSACTION_DEADLOCK:
            return RecoveryStrategy.ROLLBACK_REPLAY

        if failure_category == FailureCategory.DISK_QUOTA_EXCEEDED:
            return RecoveryStrategy.PURGE_TEMP

        if failure_category == FailureCategory.FLAKY_SERVICE_DEGRADED:
            # Deterministic policy: reset service state first; model may advise retry with backoff based on history
            return RecoveryStrategy.RESET_SERVICE

        if failure_category == FailureCategory.PARTIALLY_CORRUPTED_DATA:
            return RecoveryStrategy.SANITIZE_DATA

        if failure_category == FailureCategory.CASCADED_UPSTREAM_FAILURE:
            # Cascaded upstream failure cannot be resolved locally without upstream repair -> escalate
            return RecoveryStrategy.ESCALATE

        if failure_category == FailureCategory.VERIFICATION_FAILURE:
            params = task.parameters
            if "sanitized_data" in params:
                return RecoveryStrategy.SANITIZE_DATA
            if params.get("transient") is True or "succeed_on_attempt" in params:
                return RecoveryStrategy.RETRY
            return RecoveryStrategy.ESCALATE

        return RecoveryStrategy.ESCALATE

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
        """Evaluate guardrails in strict order and dispatch remediation if safe.

        GUARDRAIL ORDER:
          (1) Unrecoverable failure category -> immediate escalation
          (2) Non-idempotent task -> immediate escalation (never repeat unsafe state mutations)
          (3) Per-task maximum attempts exhausted -> escalation
          (4) Per-task maximum recoveries exhausted -> escalation
          (5) Shared workflow recovery budget exhausted -> escalation
          (6) Select strategy and apply remediation
        """
        # (1) Unrecoverable check
        if failure_category == FailureCategory.UNRECOVERABLE_ERROR:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Unrecoverable failure category; escalating to human operator",
                escalated=True,
            )

        # (2) Idempotency guardrail
        if not task.idempotent:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Task is marked non-idempotent; automatic retry/recovery is unsafe",
                escalated=True,
            )

        # (3) Per-task attempts guardrail
        if current_attempt >= task.max_attempts:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Per-task maximum attempts exhausted",
                escalated=True,
            )

        # (4) Per-task recoveries guardrail
        if task_recoveries_used >= task.max_recoveries:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Per-task maximum recoveries exhausted",
                escalated=True,
            )

        # (5) Workflow shared budget guardrail
        if workflow_recoveries_used >= workflow_budget:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Shared workflow recovery budget exhausted",
                escalated=True,
            )

        # (6) Strategy selection and execution
        strategy = self.determine_strategy(task, failure_category)

        if strategy == RecoveryStrategy.ESCALATE:
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=RecoveryStrategy.ESCALATE,
                applied=False,
                details="Escalating failure category to human operator",
                escalated=True,
            )

        applied = False
        details = ""

        try:
            if strategy == RecoveryStrategy.RETRY:
                applied = True
                details = f"Retrying idempotent task '{task.id}' without state mutation"

            elif strategy == RecoveryStrategy.RESET_SERVICE:
                details = simulator.reset_service(task)
                applied = True

            elif strategy == RecoveryStrategy.RESTORE_FILE:
                details = simulator.restore_file(task)
                applied = True

            elif strategy == RecoveryStrategy.SANITIZE_DATA:
                details = simulator.sanitize_data(task)
                applied = True

            elif strategy == RecoveryStrategy.ROLLBACK_REPLAY:
                details = simulator.rollback_replay(task)
                applied = True

            elif strategy == RecoveryStrategy.PURGE_TEMP:
                details = simulator.purge_temp(task)
                applied = True

        except Exception as e:
            # If a remediation action itself fails, escalate safely
            return RecoveryAttempt(
                task_id=task.id,
                recovery_number=recovery_number,
                failure_category=failure_category,
                strategy=strategy,
                applied=False,
                details=f"Remediation action failed with exception: {e}",
                escalated=True,
            )

        return RecoveryAttempt(
            task_id=task.id,
            recovery_number=recovery_number,
            failure_category=failure_category,
            strategy=strategy,
            applied=applied,
            details=details,
            escalated=False,
        )

