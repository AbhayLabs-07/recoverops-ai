"""Unit tests for Stage 2 new failure families: RATE_LIMITED, DEADLOCK, DISK_QUOTA."""

import unittest
from pathlib import Path
import tempfile
import shutil

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    Workflow,
)
from app.recovery import RecoveryManager
from app.verification import Verifier
from app.workflow import WorkflowExecutor


class TestNewFailureFamilies(unittest.TestCase):
    """Test suite verifying Stage 2 failure families and remediations."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage2_sim_")).resolve()
        self.simulator = FailureSimulator(root_dir=self.temp_dir)
        self.recovery_mgr = RecoveryManager()
        self.verifier = Verifier()
        self.executor = WorkflowExecutor(simulator=self.simulator, verifier=self.verifier)

    def tearDown(self) -> None:
        self.simulator.cleanup()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rate_limited_retry_with_logical_backoff(self) -> None:
        """Verify RATE_LIMITED action records logical backoff steps without sleeping and recovers on attempt 2."""
        task = Task(
            id="t_rate",
            name="Rate Limited API Call",
            action=TaskAction.RATE_LIMITED,
            parameters={"succeed_on_attempt": 2},
            max_attempts=3,
            max_recoveries=2,
        )
        res1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.RATE_LIMIT_EXCEEDED)
        self.assertEqual(self.simulator._logical_backoffs["t_rate"], 1)

        # Strategy classification and remediation
        strategy = self.recovery_mgr.determine_strategy(task, res1.error_code)
        self.assertEqual(strategy, RecoveryStrategy.RETRY)

        res2 = self.simulator.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertEqual(res2.error_code, FailureCategory.NONE)
        self.assertTrue(res2.return_value["rate_limit_cleared"])
        self.assertEqual(self.simulator._logical_backoffs["t_rate"], 2)

    def test_deadlock_rollback_and_replay(self) -> None:
        """Verify DEADLOCK action triggers TRANSACTION_DEADLOCK, ROLLBACK_REPLAY remediation, and succeeds on replay."""
        task = Task(
            id="t_deadlock",
            name="Database Transaction",
            action=TaskAction.DEADLOCK,
            parameters={"output": {"transaction": "committed", "replayed": True}},
            max_attempts=3,
            max_recoveries=2,
        )
        res1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.TRANSACTION_DEADLOCK)

        # Strategy mapping
        strategy = self.recovery_mgr.determine_strategy(task, res1.error_code)
        self.assertEqual(strategy, RecoveryStrategy.ROLLBACK_REPLAY)

        # Apply rollback_replay remediation
        details = self.simulator.rollback_replay(task)
        self.assertIn("rolled back and prepared for replay", details)

        res2 = self.simulator.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertTrue(res2.return_value["replayed"])

    def test_disk_quota_purge_temp_sandbox_isolation(self) -> None:
        """Verify DISK_QUOTA purges ONLY sandbox-registered temp files and never touches external paths."""
        task = Task(
            id="t_quota",
            name="Batch Processing Task",
            action=TaskAction.DISK_QUOTA,
            parameters={"output": {"disk_space_available": True, "quota_cleared": True}},
            max_attempts=3,
            max_recoveries=2,
        )
        # Register a temporary file strictly in the sandbox
        temp_file = self.simulator.register_sandbox_temp_file("tmp/cache_part1.dat", content="garbage data")
        self.assertTrue(temp_file.is_file())

        res1 = self.simulator.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.DISK_QUOTA_EXCEEDED)

        # Strategy mapping
        strategy = self.recovery_mgr.determine_strategy(task, res1.error_code)
        self.assertEqual(strategy, RecoveryStrategy.PURGE_TEMP)

        # Apply purge_temp
        purge_msg = self.simulator.purge_temp(task)
        self.assertIn("Purged 1 sandbox-registered temporary artifacts", purge_msg)
        self.assertFalse(temp_file.exists())  # Temp file removed

        res2 = self.simulator.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertTrue(res2.return_value["quota_cleared"])

    def test_workflow_healing_with_new_families(self) -> None:
        """Verify complete workflow self-healing through rate limiting, deadlock, and disk quota."""
        tasks = (
            Task(
                id="t1_rate",
                name="Fetch with Rate Limiting",
                action=TaskAction.RATE_LIMITED,
                parameters={"succeed_on_attempt": 2},
                max_attempts=3,
                max_recoveries=2,
            ),
            Task(
                id="t2_deadlock",
                name="Atomic Tx with Deadlock",
                action=TaskAction.DEADLOCK,
                parameters={"output": {"transaction": "committed", "replayed": True}},
                max_attempts=3,
                max_recoveries=2,
            ),
            Task(
                id="t3_quota",
                name="Artifact Staging with Quota",
                action=TaskAction.DISK_QUOTA,
                parameters={"output": {"disk_space_available": True, "quota_cleared": True}},
                max_attempts=3,
                max_recoveries=2,
            ),
        )
        # Register a sandbox temp file for the quota task
        self.simulator.register_sandbox_temp_file("staging/artifact.tmp", content="data")

        wf = Workflow(name="stage2_mixed_pipeline", tasks=tasks, recovery_budget=5)
        result = self.executor.execute(wf)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.status.value, "completed_with_recovery")
        self.assertEqual(result.total_recoveries_used, 3)

    def test_non_idempotent_guardrail_on_new_families(self) -> None:
        """Verify that non-idempotent tasks escalate immediately even for DEADLOCK or RATE_LIMITED."""
        task = Task(
            id="t_unsafe_deadlock",
            name="Non-idempotent DB write",
            action=TaskAction.DEADLOCK,
            idempotent=False,
        )
        wf = Workflow(name="unsafe_deadlock_wf", tasks=(task,), recovery_budget=5)
        result = self.executor.execute(wf)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.status.value, "escalated")
        self.assertEqual(result.total_recoveries_used, 0)


if __name__ == "__main__":
    unittest.main()
