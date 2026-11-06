"""Unit tests for relational SQLite audit persistence and cleanup."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    Task,
    TaskAction,
    VerificationResult,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
)
from app.storage import Storage, StorageError


class TestStorage(unittest.TestCase):
    """Test relational database schema, transactions, and Windows-safe cleanup."""

    def test_ephemeral_database_creation_and_cleanup(self) -> None:
        """Scenario 14: Ephemeral storage cleans up without Windows file locking issues."""
        storage = Storage(None)
        db_path = storage.db_path
        self.assertTrue(db_path.exists())

        # Insert record to verify DB is functioning
        t = Task(id="t1", name="T1", action=TaskAction.SUCCESS)
        wf = Workflow(name="test_ephemeral", tasks=(t,))
        wf_id = storage.create_workflow_run(wf)
        self.assertIsInstance(wf_id, int)

        # Close storage and ensure temp file is unlinked cleanly
        storage.close()
        self.assertFalse(db_path.exists())

    def test_persistent_database_preserved(self) -> None:
        """Verify persistent storage retains file after closing."""
        temp_dir = tempfile.mkdtemp()
        db_file = Path(temp_dir) / "persistent_test.db"
        storage = Storage(db_file)
        self.assertTrue(db_file.exists())

        t = Task(id="t1", name="T1", action=TaskAction.SUCCESS)
        wf = Workflow(name="test_persistent", tasks=(t,))
        wf_id = storage.create_workflow_run(wf)

        storage.close()
        self.assertTrue(db_file.exists())

        # Reopen and verify data persisted
        storage_reopen = Storage(db_file)
        runs = storage_reopen.get_workflow_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["id"], wf_id)
        storage_reopen.close()

        # Clean up temp test directory
        try:
            db_file.unlink()
            Path(temp_dir).rmdir()
        except OSError:
            pass

    def test_foreign_key_enforcement(self) -> None:
        """Verify SQLite foreign key constraints are enforced."""
        with Storage(None) as storage:
            exec_bad = ExecutionResult(
                "t1", 1, TaskAction.SUCCESS, {}, False, FailureCategory.NONE, None
            )
            with self.assertRaises(sqlite3.IntegrityError):
                storage.record_execution(99999, "Nonexistent Workflow", exec_bad)

    def test_audit_lifecycle_records(self) -> None:
        """Verify complete audit trail recording and retrieval across all tables."""
        with Storage(None) as storage:
            t = Task(id="t1", name="Task 1", action=TaskAction.SUCCESS)
            wf = Workflow(name="full_audit_workflow", tasks=(t,), recovery_budget=3)
            wf_id = storage.create_workflow_run(wf)

            # Record execution
            ex = ExecutionResult(
                task_id="t1",
                attempt_number=1,
                action=TaskAction.SUCCESS,
                return_value={"records": 50},
                exception_raised=False,
                error_code=FailureCategory.NONE,
                error_message=None,
            )
            storage.record_execution(wf_id, t.name, ex)

            # Record verification
            vr = VerificationResult(
                task_id="t1",
                attempt_number=1,
                criterion_type=CriterionType.OUTPUT_CONTAINS,
                passed=True,
                reason="Records extracted",
                observed_value={"records": 50},
            )
            storage.record_verification(wf_id, vr)

            # Record recovery
            ra = RecoveryAttempt(
                task_id="t1",
                recovery_number=1,
                failure_category=FailureCategory.NONE,
                strategy=RecoveryStrategy.RETRY,
                applied=True,
                details="No action needed",
                escalated=False,
            )
            storage.record_recovery(wf_id, ra)

            # Complete workflow
            result = WorkflowResult(
                workflow_name=wf.name,
                status=WorkflowStatus.COMPLETED,
                succeeded=True,
                escalated=False,
                escalation_reason=None,
                total_recoveries_used=0,
                recovery_budget=wf.recovery_budget,
                task_outcomes=(),
                timeline=("started", "ended"),
            )
            storage.complete_workflow_run(wf_id, result)

            # Query back
            runs = storage.get_workflow_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            self.assertEqual(runs[0]["succeeded"], 1)

            executions = storage.get_executions(wf_id)
            self.assertEqual(len(executions), 1)
            self.assertEqual(executions[0]["task_id"], "t1")
            self.assertIn("records", executions[0]["return_value_json"])

            verifications = storage.get_verifications(wf_id)
            self.assertEqual(len(verifications), 1)
            self.assertEqual(verifications[0]["passed"], 1)

            recoveries = storage.get_recoveries(wf_id)
            self.assertEqual(len(recoveries), 1)
            self.assertEqual(recoveries[0]["strategy"], "retry")

    def test_complete_nonexistent_workflow_raises_error(self) -> None:
        """Verify update of nonexistent workflow raises StorageError."""
        with Storage(None) as storage:
            result = WorkflowResult(
                workflow_name="ghost",
                status=WorkflowStatus.FAILED,
                succeeded=False,
                escalated=False,
                escalation_reason=None,
                total_recoveries_used=0,
                recovery_budget=1,
                task_outcomes=(),
                timeline=(),
            )
            with self.assertRaises(StorageError):
                storage.complete_workflow_run(99999, result)


if __name__ == "__main__":
    unittest.main()

