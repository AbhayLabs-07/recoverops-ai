"""SQLite persistence tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.models import CriterionType, SuccessCriteria, Task, TaskAction, Workflow, WorkflowStatus
from app.storage import Storage
from app.workflow import WorkflowExecutor


class StorageTests(unittest.TestCase):
    def test_workflow_events_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "audit.sqlite3"
            task = Task(
                "persisted",
                "Persisted task",
                TaskAction.TRANSIENT_SERVICE,
                SuccessCriteria(CriterionType.SERVICE_READY),
                parameters={"failures_before_success": 1},
                max_attempts=2,
                max_recoveries=1,
            )
            workflow = Workflow("stored workflow", (task,))
            with Storage(database) as storage:
                result = WorkflowExecutor(storage=storage).execute(workflow)
                workflow_row = storage.fetch_workflow(workflow.id)
                executions = storage.fetch_task_outcomes(workflow.id)
                recoveries = storage.fetch_recovery_attempts(workflow.id)
                verifications = storage.fetch_verifications(workflow.id)

            self.assertEqual(result.status, WorkflowStatus.SUCCEEDED)
            self.assertIsNotNone(workflow_row)
            self.assertEqual(workflow_row["status"], WorkflowStatus.SUCCEEDED.value)
            self.assertEqual(workflow_row["recoveries_used"], 1)
            self.assertEqual(len(executions), 2)
            self.assertEqual(len(recoveries), 1)
            self.assertEqual(len(verifications), 1)
            self.assertEqual(executions[0]["failure_kind"], "transient_service")
            self.assertEqual(recoveries[0]["strategy"], "reset_service")
            self.assertEqual(json.loads(verifications[0]["evidence_json"])["service_ready"], True)

    def test_parameterized_storage_round_trips_special_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "special.sqlite3"
            task = Task(
                "quote-task",
                "Task with 'quotes' and ?",
                TaskAction.UNRECOVERABLE,
                SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
            )
            workflow = Workflow("workflow's name", (task,))
            with Storage(database) as storage:
                WorkflowExecutor(storage=storage).execute(workflow)
                row = storage.fetch_workflow(workflow.id)

            self.assertEqual(row["name"], "workflow's name")
            self.assertEqual(storage_path_exists(database), True)

    def test_unknown_count_table_is_rejected(self) -> None:
        with Storage(":memory:") as storage:
            with self.assertRaises(ValueError):
                storage.count("workflow_executions; DROP TABLE workflow_executions", "x")

    def test_unexpected_recording_error_finalizes_workflow_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "error.sqlite3"
            circular: list[object] = []
            circular.append(circular)
            task = Task(
                "bad-output",
                "Circular output",
                TaskAction.SUCCESS,
                SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
                parameters={"output": circular},
            )
            workflow = Workflow("recording error", (task,))
            with Storage(database) as storage:
                with self.assertRaises(ValueError):
                    WorkflowExecutor(storage=storage).execute(workflow)
                self.assertEqual(storage.fetch_workflow(workflow.id)["status"], WorkflowStatus.FAILED.value)


def storage_path_exists(path: Path) -> bool:
    """Keep the assertion readable without relying on a closed Storage object."""

    return path.exists()


if __name__ == "__main__":
    unittest.main()
