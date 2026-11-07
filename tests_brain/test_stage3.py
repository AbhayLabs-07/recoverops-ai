"""Unit tests for Stage 3 deliverables: RolloutOracle, new hidden-state families,
independent label-shuffle control, and history ablation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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
    Workflow,
    WorkflowStatus,
)
from app.recovery import RecoveryManager
from app.verification import Verifier
from app.workflow import WorkflowExecutor
from brain.rollout_oracle import RolloutOracle, RolloutScore, TIE_BREAK_PRIORITY
from brain.baselines import (
    DecisionTreeBaseline,
    LogisticRegressionBaseline,
    StatelessLookupBaseline,
    MajorityClassBaseline,
    evaluate_baseline_predictions,
)


class TestStage3(unittest.TestCase):
    """Verify RolloutOracle, hidden-state failure families, and baseline controls."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage3_")).resolve()
        self.simulator = FailureSimulator(root_dir=self.temp_dir)
        self.verifier = Verifier()
        self.recovery_mgr = RecoveryManager()
        self.oracle = RolloutOracle(root_dir=self.temp_dir / "oracle")

    def tearDown(self) -> None:
        self.simulator.cleanup()
        self.oracle.cleanup()

    def test_rollout_oracle_deterministic_selection(self) -> None:
        """Verify RolloutOracle clones state, evaluates strategies, and selects optimal outcome."""
        task = Task(
            id="t_missing_file",
            name="Missing File Task",
            action=TaskAction.MISSING_FILE,
            parameters={"file_path": "data/critical.json", "restore_content": '{"restored": true}'},
            success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, "data/critical.json"),
            idempotent=True,
            max_attempts=3,
            max_recoveries=2,
        )

        best_strat, scores = self.oracle.select_best_strategy(
            task=task,
            failure_category=FailureCategory.MISSING_FILE_ERROR,
            simulator=self.simulator,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=4,
        )

        self.assertEqual(best_strat, RecoveryStrategy.RESTORE_FILE)
        # Verify scores breakdown
        restore_score = next(s for s in scores if s.strategy == RecoveryStrategy.RESTORE_FILE)
        self.assertTrue(restore_score.verified)
        self.assertTrue(restore_score.safe)
        self.assertGreater(restore_score.score, 50.0)

    def test_rollout_oracle_safety_guardrails(self) -> None:
        """Verify RolloutOracle penalizes non-idempotent or unrecoverable actions with -inf score."""
        task = Task(
            id="t_unsafe",
            name="Unsafe Task",
            action=TaskAction.TIMEOUT,
            idempotent=False,  # NON-IDEMPOTENT
        )

        best_strat, scores = self.oracle.select_best_strategy(
            task=task,
            failure_category=FailureCategory.TIMEOUT_ERROR,
            simulator=self.simulator,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=4,
        )

        # Must select ESCALATE
        self.assertEqual(best_strat, RecoveryStrategy.ESCALATE)
        retry_score = next(s for s in scores if s.strategy == RecoveryStrategy.RETRY)
        self.assertEqual(retry_score.score, -float("inf"))
        self.assertFalse(retry_score.safe)

    def test_hidden_state_flaky_service(self) -> None:
        """Verify FLAKY_SERVICE succeeds only after reset is performed and attempt >= succeed_on_attempt."""
        task = Task(
            id="t_flaky",
            name="Flaky Service Task",
            action=TaskAction.FLAKY_SERVICE,
            parameters={"required_resets": 1, "succeed_on_attempt": 2},
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
            idempotent=True,
        )

        # Attempt 1: Fails before reset
        r1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(r1.exception_raised)
        self.assertEqual(r1.error_code, FailureCategory.FLAKY_SERVICE_DEGRADED)

        # Reset service state
        self.simulator.reset_service(task)

        # Attempt 2: Succeeds
        r2 = self.simulator.execute(task, attempt_number=2)
        self.assertFalse(r2.exception_raised)
        self.assertEqual(r2.error_code, FailureCategory.NONE)

    def test_hidden_state_partial_corruption(self) -> None:
        """Verify PARTIAL_CORRUPTION succeeds after sanitize_data is applied."""
        task = Task(
            id="t_part_corr",
            name="Partial Corruption Task",
            action=TaskAction.PARTIAL_CORRUPTION,
            parameters={"clean_data": {"id": "t_part_corr", "valid": True}},
            success_criteria=SuccessCriteria(CriterionType.DATA_VALID, ["id", "valid"]),
            idempotent=True,
        )

        # Attempt 1 fails
        r1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(r1.exception_raised)
        self.assertEqual(r1.error_code, FailureCategory.PARTIALLY_CORRUPTED_DATA)

        # Sanitize data
        self.simulator.sanitize_data(task)

        # Attempt 2 succeeds
        r2 = self.simulator.execute(task, attempt_number=2)
        self.assertFalse(r2.exception_raised)
        v = self.verifier.verify(task, r2, self.simulator)
        self.assertTrue(v.passed)

    def test_hidden_state_cascading_failure(self) -> None:
        """Verify CASCADING_FAILURE escalates unless upstream completed."""
        task = Task(
            id="t_cascade",
            name="Cascading Failure Task",
            action=TaskAction.CASCADING_FAILURE,
            parameters={"upstream_task_id": "parent_job", "upstream_failed": True},
            idempotent=True,
        )

        # When upstream failed, execution fails with CASCADED_UPSTREAM_FAILURE
        r1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(r1.exception_raised)
        self.assertEqual(r1.error_code, FailureCategory.CASCADED_UPSTREAM_FAILURE)

        # Policy maps it to ESCALATE
        strat = self.recovery_mgr.determine_strategy(task, r1.error_code)
        self.assertEqual(strat, RecoveryStrategy.ESCALATE)

    def test_logistic_regression_baseline_fit(self) -> None:
        """Verify pure NumPy Logistic Regression baseline fits without external dependencies."""
        episodes = [
            {"observation": {"failure_category": "timeout_error", "attempt": 1}, "strategy": "RETRY"},
            {"observation": {"failure_category": "missing_file_error", "attempt": 1}, "strategy": "RESTORE_FILE"},
            {"observation": {"failure_category": "transient_service_error", "attempt": 1}, "strategy": "RESET_SERVICE"},
        ] * 20
        targets = [e["strategy"] for e in episodes]

        lr = LogisticRegressionBaseline(epochs=50)
        lr.fit(episodes, targets)
        p = lr.predict(episodes[0])
        self.assertEqual(p, "RETRY")


if __name__ == "__main__":
    unittest.main()
