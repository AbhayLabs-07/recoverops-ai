"""Unit tests for failure classification, strategy selection, and guardrail order."""

from __future__ import annotations

import unittest

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


class TestRecovery(unittest.TestCase):
    """Test classification logic and strict guardrail ordering in RecoveryManager."""

    def setUp(self) -> None:
        self.sim = FailureSimulator()
        self.rm = RecoveryManager()

    def tearDown(self) -> None:
        self.sim.cleanup()

    def test_classification_mapping(self) -> None:
        """Verify failure classification for execution exceptions and verification failures."""
        # Execution exception
        ex = ExecutionResult(
            "t1", 1, TaskAction.TRANSIENT_SERVICE, None, True, FailureCategory.TRANSIENT_SERVICE_ERROR, "fail"
        )
        self.assertEqual(self.rm.classify_failure(ex, None), FailureCategory.TRANSIENT_SERVICE_ERROR)

        # Verification failures
        ok_exec = ExecutionResult("t1", 1, TaskAction.SUCCESS, {}, False, FailureCategory.NONE, None)

        v_data = VerificationResult("t1", 1, CriterionType.DATA_VALID, False, "invalid")
        self.assertEqual(self.rm.classify_failure(ok_exec, v_data), FailureCategory.INVALID_DATA_ERROR)

        v_file = VerificationResult("t1", 1, CriterionType.FILE_EXISTS, False, "missing")
        self.assertEqual(self.rm.classify_failure(ok_exec, v_file), FailureCategory.MISSING_FILE_ERROR)

        v_svc = VerificationResult("t1", 1, CriterionType.SERVICE_READY, False, "offline")
        self.assertEqual(self.rm.classify_failure(ok_exec, v_svc), FailureCategory.TRANSIENT_SERVICE_ERROR)

        v_out = VerificationResult("t1", 1, CriterionType.OUTPUT_EQUALS, False, "mismatch")
        self.assertEqual(self.rm.classify_failure(ok_exec, v_out), FailureCategory.VERIFICATION_FAILURE)

    def test_guardrail_order_1_unrecoverable(self) -> None:
        """Guardrail 1: Unrecoverable category escalates first even if budget remains."""
        task = Task(id="t1", name="T", action=TaskAction.UNRECOVERABLE, idempotent=True, max_attempts=5)
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.UNRECOVERABLE_ERROR,
            simulator=self.sim,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=5,
            recovery_number=1,
        )
        self.assertTrue(rec.escalated)
        self.assertFalse(rec.applied)
        self.assertEqual(rec.strategy, RecoveryStrategy.ESCALATE)
        self.assertEqual(rec.details, "Unrecoverable failure category; escalating to human operator")

    def test_guardrail_order_2_non_idempotent(self) -> None:
        """Scenario 11: idempotent=False with recoverable failure escalates immediately."""
        task = Task(
            id="t1",
            name="Non-idempotent Task",
            action=TaskAction.TRANSIENT_SERVICE,
            idempotent=False,
            max_attempts=5,
            max_recoveries=3,
        )
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TRANSIENT_SERVICE_ERROR,
            simulator=self.sim,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=5,
            recovery_number=1,
        )
        self.assertTrue(rec.escalated)
        self.assertFalse(rec.applied)
        self.assertEqual(rec.strategy, RecoveryStrategy.ESCALATE)
        self.assertEqual(
            rec.details, "Task is marked non-idempotent; automatic retry/recovery is unsafe"
        )

    def test_guardrail_order_3_task_max_attempts(self) -> None:
        """Scenario 7: max_attempts reached causes escalation."""
        task = Task(id="t1", name="T", action=TaskAction.TIMEOUT, max_attempts=2, max_recoveries=5)
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TIMEOUT_ERROR,
            simulator=self.sim,
            current_attempt=2,  # Already at max_attempts (2)
            task_recoveries_used=1,
            workflow_recoveries_used=1,
            workflow_budget=5,
            recovery_number=2,
        )
        self.assertTrue(rec.escalated)
        self.assertFalse(rec.applied)
        self.assertEqual(rec.details, "Per-task maximum attempts exhausted")

    def test_guardrail_order_4_task_max_recoveries(self) -> None:
        """Scenario 8: max_recoveries reached causes escalation."""
        task = Task(id="t1", name="T", action=TaskAction.TIMEOUT, max_attempts=10, max_recoveries=1)
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TIMEOUT_ERROR,
            simulator=self.sim,
            current_attempt=2,
            task_recoveries_used=1,  # Already reached max_recoveries (1)
            workflow_recoveries_used=1,
            workflow_budget=5,
            recovery_number=2,
        )
        self.assertTrue(rec.escalated)
        self.assertFalse(rec.applied)
        self.assertEqual(rec.details, "Per-task maximum recoveries exhausted")

    def test_guardrail_order_5_workflow_budget(self) -> None:
        """Scenario 9: Shared workflow budget exhausted causes escalation."""
        task = Task(id="t1", name="T", action=TaskAction.TIMEOUT, max_attempts=5, max_recoveries=5)
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TIMEOUT_ERROR,
            simulator=self.sim,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=2,  # Already reached workflow budget (2)
            workflow_budget=2,
            recovery_number=1,
        )
        self.assertTrue(rec.escalated)
        self.assertFalse(rec.applied)
        self.assertEqual(rec.details, "Shared workflow recovery budget exhausted")

    def test_strategy_application_and_budget_neutrality(self) -> None:
        """Verify successful strategy dispatch and check that RecoveryManager does not alter counters."""
        task = Task(id="svc", name="Service", action=TaskAction.TRANSIENT_SERVICE)
        rec = self.rm.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TRANSIENT_SERVICE_ERROR,
            simulator=self.sim,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=3,
            recovery_number=1,
        )
        self.assertFalse(rec.escalated)
        self.assertTrue(rec.applied)
        self.assertEqual(rec.strategy, RecoveryStrategy.RESET_SERVICE)
        self.assertTrue(self.sim.is_service_ready("svc"))


if __name__ == "__main__":
    unittest.main()

