"""End-to-end workflow execution tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    FailureKind,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)
from app.verification import Verifier
from app.workflow import WorkflowExecutor


def task(
    task_id: str,
    action: TaskAction,
    criterion: CriterionType,
    *,
    expected: object = None,
    parameters: dict[str, object] | None = None,
    max_attempts: int = 3,
    max_recoveries: int = 2,
    idempotent: bool = True,
) -> Task:
    return Task(
        id=task_id,
        name=task_id,
        action=action,
        parameters=parameters or {},
        success_criteria=SuccessCriteria(criterion, expected),
        max_attempts=max_attempts,
        max_recoveries=max_recoveries,
        idempotent=idempotent,
    )


class WorkflowExecutionTests(unittest.TestCase):
    def test_successful_workflow_requires_verification(self) -> None:
        workflow = Workflow(
            "success",
            (
                task(
                    "ready",
                    TaskAction.SUCCESS,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"ok": True},
                    parameters={"output": {"ok": True}},
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)

        self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.task_outcomes["ready"].status, TaskStatus.SUCCEEDED)
        self.assertEqual(len(result.task_outcomes["ready"].verifications), 1)

    def test_transient_failure_is_recovered(self) -> None:
        workflow = Workflow(
            "transient",
            (
                task(
                    "service",
                    TaskAction.TRANSIENT_SERVICE,
                    CriterionType.SERVICE_READY,
                    parameters={"failures_before_success": 1},
                    max_attempts=2,
                    max_recoveries=1,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)
        outcome = result.task_outcomes["service"]

        self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
        self.assertEqual(outcome.executions[0].failure_kind, FailureKind.TRANSIENT_SERVICE)
        self.assertEqual(outcome.recoveries[0].strategy, RecoveryStrategy.RESET_SERVICE)
        self.assertEqual(len(outcome.executions), 2)

    def test_timeout_is_retried_and_verified(self) -> None:
        workflow = Workflow(
            "timeout",
            (
                task(
                    "slow-service",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"completed": True},
                    max_attempts=2,
                    max_recoveries=1,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)
        outcome = result.task_outcomes["slow-service"]

        self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
        self.assertEqual(outcome.executions[0].failure_kind, FailureKind.TIMEOUT)
        self.assertEqual(outcome.recoveries[0].strategy, RecoveryStrategy.RETRY)
        self.assertTrue(outcome.verifications[-1].succeeded)

    def test_missing_file_is_restored_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = Workflow(
                "file-recovery",
                (
                    task(
                        "config-file",
                        TaskAction.MISSING_FILE,
                        CriterionType.FILE_EXISTS,
                        expected="config.txt",
                        parameters={
                            "path": "config.txt",
                            "recovery_content": "restored config",
                        },
                        max_attempts=2,
                        max_recoveries=1,
                    ),
                ),
            )

            result = WorkflowExecutor(
                simulator=FailureSimulator(root), verifier=Verifier(root)
            ).execute(workflow)

            self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
            self.assertEqual((root / "config.txt").read_text(encoding="utf-8"), "restored config")
            self.assertEqual(result.task_outcomes["config-file"].recoveries[0].strategy, RecoveryStrategy.RESTORE_FILE)

    def test_invalid_data_is_sanitized_after_failed_verification(self) -> None:
        workflow = Workflow(
            "data-recovery",
            (
                task(
                    "payload",
                    TaskAction.INVALID_DATA,
                    CriterionType.DATA_VALID,
                    max_attempts=2,
                    max_recoveries=1,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)
        outcome = result.task_outcomes["payload"]

        self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
        self.assertEqual(outcome.verifications[0].status.value, "failed")
        self.assertEqual(outcome.recoveries[0].strategy, RecoveryStrategy.SANITIZE_DATA)
        self.assertTrue(outcome.verifications[-1].succeeded)

    def test_failed_verification_never_becomes_success(self) -> None:
        workflow = Workflow(
            "bad-verification",
            (
                task(
                    "wrong-output",
                    TaskAction.SUCCESS,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"ok": True},
                    parameters={"output": {"ok": False}},
                    max_attempts=1,
                    max_recoveries=0,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)

        self.assertEqual(result.status, WorkflowStatus.FAILED)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.task_outcomes["wrong-output"].status, TaskStatus.FAILED)
        self.assertFalse(result.task_outcomes["wrong-output"].verifications[-1].succeeded)

    def test_unrecoverable_error_escalates_to_human(self) -> None:
        workflow = Workflow(
            "escalate",
            (
                task(
                    "fatal",
                    TaskAction.UNRECOVERABLE,
                    CriterionType.OUTPUT_NOT_EMPTY,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)
        outcome = result.task_outcomes["fatal"]

        self.assertEqual(result.status, WorkflowStatus.ESCALATED)
        self.assertEqual(outcome.status, TaskStatus.ESCALATED)
        self.assertEqual(outcome.recoveries[0].strategy, RecoveryStrategy.ESCALATE)

    def test_retry_limit_stops_persistent_timeout(self) -> None:
        workflow = Workflow(
            "retry-limit",
            (
                task(
                    "persistent-timeout",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"completed": True},
                    parameters={"never_recover": True},
                    max_attempts=3,
                    max_recoveries=5,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)
        outcome = result.task_outcomes["persistent-timeout"]

        self.assertEqual(result.status, WorkflowStatus.RECOVERY_EXHAUSTED)
        self.assertEqual(len(outcome.executions), 3)
        self.assertEqual(len(outcome.recoveries), 2)

    def test_workflow_recovery_budget_is_shared_and_bounded(self) -> None:
        workflow = Workflow(
            "budget",
            (
                task(
                    "first",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"completed": True},
                    parameters={"never_recover": True},
                    max_attempts=3,
                    max_recoveries=5,
                ),
                task(
                    "second",
                    TaskAction.SUCCESS,
                    CriterionType.OUTPUT_NOT_EMPTY,
                    parameters={"output": {"ok": True}},
                ),
            ),
            recovery_budget=1,
        )

        result = WorkflowExecutor().execute(workflow)

        self.assertEqual(result.status, WorkflowStatus.RECOVERY_EXHAUSTED)
        self.assertEqual(len(result.task_outcomes["first"].recoveries), 1)
        self.assertEqual(result.task_outcomes["second"].status, TaskStatus.PENDING)

    def test_non_idempotent_task_is_not_repeated_blindly(self) -> None:
        workflow = Workflow(
            "non-idempotent",
            (
                task(
                    "charge-card",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_NOT_EMPTY,
                    idempotent=False,
                ),
            ),
        )

        result = WorkflowExecutor().execute(workflow)

        self.assertEqual(result.status, WorkflowStatus.ESCALATED)
        self.assertEqual(len(result.task_outcomes["charge-card"].executions), 1)

    def test_duplicate_task_ids_and_invalid_configuration_are_rejected(self) -> None:
        first = task("same", TaskAction.SUCCESS, CriterionType.OUTPUT_NOT_EMPTY)
        second = task("same", TaskAction.SUCCESS, CriterionType.OUTPUT_NOT_EMPTY)
        with self.assertRaises(ValueError):
            Workflow("duplicates", (first, second))
        with self.assertRaises(ValueError):
            Task("bad", "bad", TaskAction.SUCCESS, SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY), max_attempts=0)
        with self.assertRaises(ValueError):
            Task("float", "float", TaskAction.SUCCESS, SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY), max_attempts=1.5)
        with self.assertRaises(ValueError):
            Workflow("float-budget", (first,), recovery_budget=0.5)
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.OUTPUT_EQUALS)

    def test_same_task_id_in_separate_workflows_has_isolated_simulator_state(self) -> None:
        first = Workflow(
            "first-run",
            (
                task(
                    "shared-id",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"completed": True},
                    max_attempts=2,
                    max_recoveries=1,
                ),
            ),
        )
        second = Workflow(
            "second-run",
            (
                task(
                    "shared-id",
                    TaskAction.TIMEOUT,
                    CriterionType.OUTPUT_EQUALS,
                    expected={"completed": True},
                    max_attempts=2,
                    max_recoveries=1,
                ),
            ),
        )
        executor = WorkflowExecutor(simulator=FailureSimulator())

        first_result = executor.execute(first)
        second_result = executor.execute(second)

        self.assertEqual(len(first_result.task_outcomes["shared-id"].recoveries), 1)
        self.assertEqual(len(second_result.task_outcomes["shared-id"].recoveries), 1)


if __name__ == "__main__":
    unittest.main()
