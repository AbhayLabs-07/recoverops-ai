"""Local SQLite persistence for workflow audit records."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .models import (
    ExecutionResult,
    RecoveryAttempt,
    VerificationResult,
    Workflow,
    WorkflowStatus,
)


class Storage:
    """Persist workflow events using parameterized SQLite statements."""

    def __init__(self, database_path: str | Path = "recoverops.db") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_executions (
                workflow_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                recovery_budget INTEGER NOT NULL,
                recoveries_used INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS task_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL REFERENCES workflow_executions(workflow_id),
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                executor_succeeded INTEGER NOT NULL,
                failure_kind TEXT NOT NULL,
                output_json TEXT,
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recovery_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL REFERENCES workflow_executions(workflow_id),
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                reason TEXT NOT NULL,
                succeeded INTEGER NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL REFERENCES workflow_executions(workflow_id),
                task_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                criterion TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                error TEXT,
                verified_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_task_outcomes_workflow
                ON task_outcomes(workflow_id, task_id);
            CREATE INDEX IF NOT EXISTS idx_recovery_attempts_workflow
                ON recovery_attempts(workflow_id, task_id);
            CREATE INDEX IF NOT EXISTS idx_verification_results_workflow
                ON verification_results(workflow_id, task_id);
            """
        )
        self.connection.commit()

    def begin_workflow(self, workflow: Workflow, started_at: datetime) -> None:
        """Create the parent audit row for a workflow run."""

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO workflow_executions
                    (workflow_id, name, status, recovery_budget, recoveries_used, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow.id,
                    workflow.name,
                    WorkflowStatus.RUNNING.value,
                    workflow.recovery_budget,
                    0,
                    _timestamp(started_at),
                ),
            )

    def finish_workflow(
        self,
        workflow_id: str,
        status: WorkflowStatus,
        recoveries_used: int,
        finished_at: datetime,
    ) -> None:
        """Commit the final workflow state and recovery count."""

        with self.connection:
            self.connection.execute(
                """
                UPDATE workflow_executions
                SET status = ?, recoveries_used = ?, finished_at = ?
                WHERE workflow_id = ?
                """,
                (status.value, recoveries_used, _timestamp(finished_at), workflow_id),
            )

    def record_execution(self, result: ExecutionResult) -> None:
        """Persist one direct action result."""

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO task_outcomes
                    (workflow_id, task_id, attempt, executor_succeeded,
                     failure_kind, output_json, error, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.workflow_id,
                    result.task_id,
                    result.attempt,
                    int(result.executor_succeeded),
                    result.failure_kind.value,
                    _json(result.output),
                    result.error,
                    _timestamp(result.started_at),
                    _timestamp(result.finished_at),
                ),
            )

    def record_recovery(self, attempt: RecoveryAttempt) -> None:
        """Persist one recovery or escalation decision."""

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO recovery_attempts
                    (workflow_id, task_id, attempt, strategy, reason,
                     succeeded, message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.workflow_id,
                    attempt.task_id,
                    attempt.attempt,
                    attempt.strategy.value,
                    attempt.reason,
                    int(attempt.succeeded),
                    attempt.message,
                    _timestamp(attempt.timestamp),
                ),
            )

    def record_verification(self, result: VerificationResult) -> None:
        """Persist an independent verification result."""

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO verification_results
                    (workflow_id, task_id, attempt, status, criterion,
                     evidence_json, error, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.workflow_id,
                    result.task_id,
                    result.attempt,
                    result.status.value,
                    result.criterion.value,
                    _json(result.evidence),
                    result.error,
                    _timestamp(result.verified_at),
                ),
            )

    def fetch_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        """Return one workflow row as a plain dictionary."""

        row = self.connection.execute(
            "SELECT * FROM workflow_executions WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def fetch_task_outcomes(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return all persisted direct task executions for a workflow."""

        return self._fetch_all(
            "SELECT * FROM task_outcomes WHERE workflow_id = ? ORDER BY id", workflow_id
        )

    def fetch_recovery_attempts(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return all persisted recovery attempts for a workflow."""

        return self._fetch_all(
            "SELECT * FROM recovery_attempts WHERE workflow_id = ? ORDER BY id", workflow_id
        )

    def fetch_verifications(self, workflow_id: str) -> list[dict[str, Any]]:
        """Return all persisted independent checks for a workflow."""

        return self._fetch_all(
            "SELECT * FROM verification_results WHERE workflow_id = ? ORDER BY id", workflow_id
        )

    def count(self, table: str, workflow_id: str) -> int:
        """Return a record count for a known audit table.

        The table name is selected from a closed allow-list rather than
        interpolated from untrusted input.
        """

        allowed = {
            "task_outcomes",
            "recovery_attempts",
            "verification_results",
        }
        if table not in allowed:
            raise ValueError("unsupported audit table")
        row = self.connection.execute(
            f"SELECT COUNT(*) AS count FROM {table} WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        return int(row["count"])

    def _fetch_all(self, query: str, workflow_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(query, (workflow_id,)).fetchall()]

    def close(self) -> None:
        """Close the SQLite connection."""

        self.connection.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def _timestamp(value: datetime | date) -> str:
    return value.isoformat()


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=_json_default, sort_keys=True)


def _json_default(value: Any) -> str:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)
