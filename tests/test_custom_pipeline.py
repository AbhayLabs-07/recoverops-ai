"""Unit test suite for the custom simulated config-sync-and-publish pipeline."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.models import WorkflowStatus
from scripts.custom_pipeline import run_pipeline


class TestCustomPipeline(unittest.TestCase):
    """Test suite verifying simulated config-sync-and-publish pipeline scenarios."""

    def test_happy_scenario(self) -> None:
        """Scenario 'happy': all tasks succeed on attempt 1 without recoveries."""
        result, audit = run_pipeline(scenario="happy", recovery_budget=5)
        self.assertTrue(result.succeeded)
        self.assertFalse(result.escalated)
        self.assertEqual(result.status, WorkflowStatus.COMPLETED)
        self.assertEqual(result.total_recoveries_used, 0)
        self.assertEqual(audit["execution_records"], 5)
        self.assertEqual(audit["recovery_records"], 0)

    def test_healing_scenario(self) -> None:
        """Scenario 'healing': tasks 1-4 heal and complete within budget."""
        result, audit = run_pipeline(scenario="healing", recovery_budget=5)
        self.assertTrue(result.succeeded)
        self.assertFalse(result.escalated)
        self.assertEqual(result.status, WorkflowStatus.COMPLETED_WITH_RECOVERY)
        self.assertEqual(result.total_recoveries_used, 4)
        self.assertEqual(audit["execution_records"], 9)
        self.assertEqual(audit["recovery_records"], 4)

    def test_budget_exhaustion_scenario(self) -> None:
        """Scenario 'healing' with budget=2: exhausts shared budget on task 3 and escalates."""
        result, audit = run_pipeline(scenario="healing", recovery_budget=2)
        self.assertFalse(result.succeeded)
        self.assertTrue(result.escalated)
        self.assertEqual(result.status, WorkflowStatus.ESCALATED)
        self.assertEqual(result.total_recoveries_used, 2)
        self.assertIn("Shared workflow recovery budget exhausted", result.escalation_reason or "")
        self.assertEqual(audit["recovery_records"], 3)  # 2 applied + 1 escalation record

    def test_unsafe_non_idempotent_escalation(self) -> None:
        """Scenario 'unsafe': task 5 is non-idempotent; fault triggers immediate escalation."""
        result, audit = run_pipeline(scenario="unsafe", recovery_budget=5)
        self.assertFalse(result.succeeded)
        self.assertTrue(result.escalated)
        self.assertEqual(result.status, WorkflowStatus.ESCALATED)
        self.assertEqual(result.total_recoveries_used, 0)
        self.assertIn("Task is marked non-idempotent", result.escalation_reason or "")

    def test_determinism_across_repeated_runs(self) -> None:
        """Verify repeated execution produces identical status, recoveries, and audit counts."""
        res1, audit1 = run_pipeline(scenario="healing", recovery_budget=5)
        res2, audit2 = run_pipeline(scenario="healing", recovery_budget=5)

        self.assertEqual(res1.status, res2.status)
        self.assertEqual(res1.total_recoveries_used, res2.total_recoveries_used)
        self.assertEqual(audit1["execution_records"], audit2["execution_records"])
        self.assertEqual(audit1["recovery_records"], audit2["recovery_records"])

    def test_custom_and_temp_database_lifecycle(self) -> None:
        """Verify custom DB persistence and ephemeral DB cleanup on Windows."""
        temp_dir = tempfile.mkdtemp(prefix="custom_pipe_test_")
        target_db = Path(temp_dir) / "persistent_test.db"

        try:
            _, audit = run_pipeline(scenario="happy", recovery_budget=3, db_path=target_db)
            self.assertTrue(target_db.is_file())
            self.assertEqual(audit["execution_records"], 5)
        finally:
            if target_db.exists():
                target_db.unlink(missing_ok=True)
            if Path(temp_dir).exists():
                Path(temp_dir).rmdir()


if __name__ == "__main__":
    unittest.main()
