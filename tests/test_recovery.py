"""Recovery classification and safety tests."""

from __future__ import annotations

import unittest

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    FailureKind,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
)
from app.recovery import RecoveryManager


class RecoveryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = RecoveryManager()
        self.simulator = FailureSimulator()
        self.task = Task(
            "task",
            "Known task",
            TaskAction.SUCCESS,
            SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
        )

    def test_known_failures_map_to_predefined_strategies(self) -> None:
        expected = {
            FailureKind.TRANSIENT_SERVICE: RecoveryStrategy.RESET_SERVICE,
            FailureKind.TIMEOUT: RecoveryStrategy.RETRY,
            FailureKind.MISSING_FILE: RecoveryStrategy.RESTORE_FILE,
            FailureKind.INVALID_DATA: RecoveryStrategy.SANITIZE_DATA,
            FailureKind.UNKNOWN: RecoveryStrategy.ESCALATE,
            FailureKind.UNRECOVERABLE: RecoveryStrategy.ESCALATE,
        }

        for failure, strategy in expected.items():
            with self.subTest(failure=failure):
                self.assertEqual(self.manager.decide(self.task, failure).strategy, strategy)

    def test_non_idempotent_work_is_escalated(self) -> None:
        task = Task(
            "side-effect",
            "Non-idempotent operation",
            TaskAction.TIMEOUT,
            SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
            idempotent=False,
        )

        decision = self.manager.decide(task, FailureKind.TIMEOUT)

        self.assertEqual(decision.strategy, RecoveryStrategy.ESCALATE)
        attempt = self.manager.apply(task, "workflow", 1, FailureKind.TIMEOUT, self.simulator)
        self.assertFalse(attempt.succeeded)
        self.assertEqual(attempt.strategy, RecoveryStrategy.ESCALATE)

    def test_recovery_attempt_is_auditable(self) -> None:
        task = Task(
            "timeout",
            "Timeout operation",
            TaskAction.TIMEOUT,
            SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"completed": True}),
        )

        attempt = self.manager.apply(task, "workflow-1", 1, FailureKind.TIMEOUT, self.simulator)

        self.assertEqual(attempt.workflow_id, "workflow-1")
        self.assertEqual(attempt.task_id, "timeout")
        self.assertEqual(attempt.strategy, RecoveryStrategy.RETRY)
        self.assertTrue(attempt.succeeded)
        self.assertTrue(attempt.timestamp.tzinfo is not None)

    def test_untrusted_strategy_is_not_executed(self) -> None:
        task = Task(
            "safe",
            "Safe operation",
            TaskAction.SUCCESS,
            SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
        )

        succeeded, message = self.simulator.apply_recovery(task, RecoveryStrategy.ESCALATE)

        self.assertFalse(succeeded)
        self.assertIn("human intervention", message)


if __name__ == "__main__":
    unittest.main()
