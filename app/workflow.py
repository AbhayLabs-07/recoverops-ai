"""Workflow orchestration engine enforcing budgets, guardrails, and deterministic state transitions."""

from __future__ import annotations

import logging
from typing import Any

from app.failure_simulator import FailureSimulator
from app.models import (
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    Task,
    TaskOutcome,
    TaskStatus,
    VerificationResult,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
    _current_timestamp,
)
from app.recovery import RecoveryManager
from app.storage import Storage
from app.verification import Verifier

logger = logging.getLogger("recoverops.workflow")


class WorkflowExecutor:
    """Orchestrates deterministic task sequences with bounded self-healing.

    Architectural Invariants:
      1. Single owner of recovery counters: only WorkflowExecutor increments recovery budgets.
      2. Non-idempotent tasks are never automatically retried or recovered.
      3. Strict halt-on-escalation: no subsequent tasks run once an escalation occurs.
      4. Independent verification: absence of an exception does not imply task success.
      5. Full audit trail: all attempts, verifications, and recoveries are persisted.
    """

    def __init__(
        self,
        simulator: FailureSimulator | None = None,
        verifier: Verifier | None = None,
        recovery_manager: RecoveryManager | None = None,
        storage: Storage | None = None,
    ) -> None:
        self.simulator = simulator if simulator is not None else FailureSimulator()
        self.verifier = verifier if verifier is not None else Verifier()
        self.recovery_manager = recovery_manager if recovery_manager is not None else RecoveryManager()
        self.storage = storage if storage is not None else Storage()

    def execute(self, workflow: Workflow) -> WorkflowResult:
        """Execute workflow tasks in sequence, managing recovery budgets and audit state."""
        workflow_id = self.storage.create_workflow_run(workflow)
        shared_recoveries_used = 0
        timeline: list[str] = [
            f"[{_current_timestamp()}] Workflow '{workflow.name}' started with recovery budget {workflow.recovery_budget}"
        ]

        task_outcomes: list[TaskOutcome] = []
        workflow_escalated = False
        workflow_escalation_reason: str | None = None

        for task in workflow.tasks:
            task_executions: list[ExecutionResult] = []
            task_verifications: list[VerificationResult] = []
            task_recoveries: list[RecoveryAttempt] = []
            task_recoveries_used = 0
            attempt = 1
            task_status = TaskStatus.FAILED
            final_output: Any = None
            task_escalated = False
            task_escalation_reason: str | None = None

            timeline.append(f"[{_current_timestamp()}] Starting task '{task.id}' ({task.name})")

            while True:
                # 1. Execute task attempt within controlled exception boundary
                try:
                    execution = self.simulator.execute(task, attempt)
                except Exception as exc:
                    execution = ExecutionResult(
                        task_id=task.id,
                        attempt_number=attempt,
                        action=task.action,
                        return_value=None,
                        exception_raised=True,
                        error_code=FailureCategory.UNRECOVERABLE_ERROR,
                        error_message=f"Unhandled exception during task execution: {exc}",
                    )

                task_executions.append(execution)
                self.storage.record_execution(workflow_id, task.name, execution)
                timeline.append(
                    f"[{_current_timestamp()}] Task '{task.id}' attempt {attempt} executed: "
                    f"action={execution.action.value}, error={execution.error_code.value}"
                )

                # 2. Independently verify the outcome
                verification = self.verifier.verify(task, execution, self.simulator)
                task_verifications.append(verification)
                self.storage.record_verification(workflow_id, verification)
                timeline.append(
                    f"[{_current_timestamp()}] Task '{task.id}' attempt {attempt} verified: "
                    f"criterion={verification.criterion_type.value}, passed={verification.passed}, "
                    f"reason='{verification.reason}'"
                )

                # 3. Evaluate success condition
                if not execution.exception_raised and verification.passed:
                    task_status = (
                        TaskStatus.SUCCEEDED
                        if task_recoveries_used == 0
                        else TaskStatus.RECOVERED
                    )
                    final_output = execution.return_value
                    timeline.append(
                        f"[{_current_timestamp()}] Task '{task.id}' completed successfully with status {task_status.value}"
                    )
                    break

                # 4. Failure occurred: classify failure category
                failure_cat = self.recovery_manager.classify_failure(execution, verification)
                recovery_number = len(task_recoveries) + 1

                # 5. Evaluate recovery guardrails
                recovery_attempt = self.recovery_manager.attempt_recovery(
                    task=task,
                    failure_category=failure_cat,
                    simulator=self.simulator,
                    current_attempt=attempt,
                    task_recoveries_used=task_recoveries_used,
                    workflow_recoveries_used=shared_recoveries_used,
                    workflow_budget=workflow.recovery_budget,
                    recovery_number=recovery_number,
                )

                task_recoveries.append(recovery_attempt)
                self.storage.record_recovery(workflow_id, recovery_attempt)
                timeline.append(
                    f"[{_current_timestamp()}] Task '{task.id}' recovery #{recovery_number}: "
                    f"strategy={recovery_attempt.strategy.value}, applied={recovery_attempt.applied}, "
                    f"escalated={recovery_attempt.escalated}, details='{recovery_attempt.details}'"
                )

                # 6. Check for escalation or budget exhaustion
                if recovery_attempt.escalated or not recovery_attempt.applied:
                    task_status = TaskStatus.ESCALATED
                    task_escalated = True
                    task_escalation_reason = recovery_attempt.details
                    final_output = execution.return_value
                    timeline.append(
                        f"[{_current_timestamp()}] Task '{task.id}' escalated: {task_escalation_reason}"
                    )
                    break

                # 7. Recovery successfully applied: update single-owner counters
                task_recoveries_used += 1
                shared_recoveries_used += 1
                attempt += 1

            outcome = TaskOutcome(
                task_id=task.id,
                task_name=task.name,
                status=task_status,
                attempts_used=attempt,
                recoveries_used=task_recoveries_used,
                final_output=final_output,
                escalated=task_escalated,
                escalation_reason=task_escalation_reason,
                executions=tuple(task_executions),
                verifications=tuple(task_verifications),
                recoveries=tuple(task_recoveries),
            )
            task_outcomes.append(outcome)

            if task_escalated:
                workflow_escalated = True
                workflow_escalation_reason = (
                    f"Workflow halted: task '{task.id}' escalated. Reason: {task_escalation_reason}"
                )
                timeline.append(
                    f"[{_current_timestamp()}] Execution halted. Later tasks skipped due to escalation on task '{task.id}'"
                )
                break

        # Finalize aggregate workflow result
        if workflow_escalated:
            workflow_status = WorkflowStatus.ESCALATED
            workflow_succeeded = False
        else:
            workflow_succeeded = True
            workflow_status = (
                WorkflowStatus.COMPLETED
                if shared_recoveries_used == 0
                else WorkflowStatus.COMPLETED_WITH_RECOVERY
            )

        timeline.append(
            f"[{_current_timestamp()}] Workflow '{workflow.name}' finished with status {workflow_status.value}"
        )

        result = WorkflowResult(
            workflow_name=workflow.name,
            status=workflow_status,
            succeeded=workflow_succeeded,
            escalated=workflow_escalated,
            escalation_reason=workflow_escalation_reason,
            total_recoveries_used=shared_recoveries_used,
            recovery_budget=workflow.recovery_budget,
            task_outcomes=tuple(task_outcomes),
            timeline=tuple(timeline),
        )

        self.storage.complete_workflow_run(workflow_id, result)
        return result

