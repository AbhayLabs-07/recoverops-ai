"""Sequential workflow execution with diagnosis, recovery, and verification."""

from __future__ import annotations

from datetime import datetime

from .failure_simulator import FailureSimulator
from .models import (
    FailureKind,
    RecoveryStrategy,
    Task,
    TaskOutcome,
    TaskStatus,
    VerificationStatus,
    Workflow,
    WorkflowRunResult,
    WorkflowStatus,
    utc_now,
)
from .recovery import RecoveryManager
from .storage import Storage
from .verification import Verifier


class WorkflowExecutor:
    """Run tasks in order and stop safely at the first unresolved failure."""

    def __init__(
        self,
        simulator: FailureSimulator | None = None,
        verifier: Verifier | None = None,
        recovery_manager: RecoveryManager | None = None,
        storage: Storage | None = None,
    ) -> None:
        self.simulator = simulator or FailureSimulator()
        self.verifier = verifier or Verifier()
        self.recovery_manager = recovery_manager or RecoveryManager()
        self.storage = storage

    def execute(self, workflow: Workflow) -> WorkflowRunResult:
        """Execute a workflow and return its complete in-memory audit report."""

        started = utc_now()
        if self.storage is not None:
            self.storage.begin_workflow(workflow, started)

        timeline: list[str] = [
            f"workflow {workflow.name} ({workflow.id}) started at {started.isoformat()}"
        ]
        outcomes = {task.id: TaskOutcome(task.id) for task in workflow.tasks}
        recoveries_used = 0
        final_status = WorkflowStatus.SUCCEEDED

        try:
            for task in workflow.tasks:
                outcome, used, task_status = self._execute_task(
                    workflow,
                    task,
                    outcomes[task.id],
                    recoveries_used,
                    timeline,
                )
                outcomes[task.id] = outcome
                recoveries_used = used
                if task_status is not TaskStatus.SUCCEEDED:
                    final_status = _workflow_status(task_status)
                    break
        except Exception as exc:
            # Keep a persisted run from being left in RUNNING if an unexpected
            # integration or serialization error escapes the safe path.
            final_status = WorkflowStatus.FAILED
            timeline.append(f"workflow aborted unexpectedly: {exc}")
            raise
        finally:
            finished = utc_now()
            timeline.append(f"workflow finished: {final_status.value} at {finished.isoformat()}")
            if self.storage is not None:
                self.storage.finish_workflow(workflow.id, final_status, recoveries_used, finished)
        return WorkflowRunResult(workflow.id, final_status, outcomes, timeline)

    def _execute_task(
        self,
        workflow: Workflow,
        task: Task,
        outcome: TaskOutcome,
        recoveries_used: int,
        timeline: list[str],
    ) -> tuple[TaskOutcome, int, TaskStatus]:
        task_recoveries = 0

        for attempt in range(1, task.max_attempts + 1):
            execution = self.simulator.execute(task, workflow.id, attempt)
            outcome.executions.append(execution)
            if self.storage is not None:
                self.storage.record_execution(execution)
            if execution.executor_succeeded:
                timeline.append(
                    f"task {task.id} attempt {attempt}: action returned successfully"
                )
                verification = self.verifier.verify(task, execution)
                outcome.verifications.append(verification)
                if self.storage is not None:
                    self.storage.record_verification(verification)
                timeline.append(
                    f"task {task.id} attempt {attempt}: verification "
                    f"{verification.status.value}"
                    + (f" ({verification.error})" if verification.error else "")
                )
                if verification.status is VerificationStatus.PASSED:
                    outcome.status = TaskStatus.SUCCEEDED
                    return outcome, recoveries_used, TaskStatus.SUCCEEDED
                failure_kind = FailureKind.VERIFICATION_FAILED
                outcome.error = verification.error or "verification failed"
            else:
                failure_kind = execution.failure_kind
                outcome.error = execution.error or failure_kind.value
                timeline.append(
                    f"task {task.id} attempt {attempt}: action failed "
                    f"({failure_kind.value}) - {outcome.error}"
                )

            decision = self.recovery_manager.decide(task, failure_kind)
            if decision.strategy is RecoveryStrategy.ESCALATE:
                recovery = self.recovery_manager.apply(
                    task,
                    workflow.id,
                    attempt,
                    failure_kind,
                    self.simulator,
                )
                outcome.recoveries.append(recovery)
                if self.storage is not None:
                    self.storage.record_recovery(recovery)
                timeline.append(
                    f"task {task.id} recovery {recovery.strategy.value}: "
                    f"escalated - {recovery.message}"
                )
                outcome.status = TaskStatus.ESCALATED
                return outcome, recoveries_used, TaskStatus.ESCALATED

            if attempt >= task.max_attempts:
                outcome.status = (
                    TaskStatus.FAILED
                    if failure_kind is FailureKind.VERIFICATION_FAILED
                    else TaskStatus.RECOVERY_EXHAUSTED
                )
                timeline.append(
                    f"task {task.id}: attempt limit ({task.max_attempts}) reached; stopping safely"
                )
                return outcome, recoveries_used, outcome.status
            if task_recoveries >= task.max_recoveries:
                outcome.status = (
                    TaskStatus.FAILED
                    if failure_kind is FailureKind.VERIFICATION_FAILED
                    else TaskStatus.RECOVERY_EXHAUSTED
                )
                timeline.append(
                    f"task {task.id}: per-task recovery limit ({task.max_recoveries}) reached"
                )
                return outcome, recoveries_used, outcome.status
            if recoveries_used >= workflow.recovery_budget:
                outcome.status = TaskStatus.RECOVERY_EXHAUSTED
                timeline.append(
                    f"task {task.id}: workflow recovery budget ({workflow.recovery_budget}) exhausted"
                )
                return outcome, recoveries_used, TaskStatus.RECOVERY_EXHAUSTED

            recovery = self.recovery_manager.apply(
                task,
                workflow.id,
                attempt,
                failure_kind,
                self.simulator,
            )
            outcome.recoveries.append(recovery)
            if self.storage is not None:
                self.storage.record_recovery(recovery)
            timeline.append(
                f"task {task.id} recovery {recovery.strategy.value}: "
                f"{'succeeded' if recovery.succeeded else 'failed'} - {recovery.message}"
            )
            task_recoveries += 1
            recoveries_used += 1
            if not recovery.succeeded:
                # The automatic action was bounded and could not repair the
                # known condition. Stop without another blind retry; the
                # operator can inspect the recorded attempt and intervene.
                outcome.status = TaskStatus.RECOVERY_EXHAUSTED
                outcome.error = recovery.message
                return outcome, recoveries_used, TaskStatus.RECOVERY_EXHAUSTED

        # The range always returns, but this defensive branch keeps the final
        # state explicit if the loop is changed later.
        outcome.status = TaskStatus.RECOVERY_EXHAUSTED
        return outcome, recoveries_used, TaskStatus.RECOVERY_EXHAUSTED


def _workflow_status(task_status: TaskStatus) -> WorkflowStatus:
    if task_status is TaskStatus.ESCALATED:
        return WorkflowStatus.ESCALATED
    if task_status is TaskStatus.RECOVERY_EXHAUSTED:
        return WorkflowStatus.RECOVERY_EXHAUSTED
    if task_status is TaskStatus.FAILED:
        return WorkflowStatus.FAILED
    return WorkflowStatus.FAILED
