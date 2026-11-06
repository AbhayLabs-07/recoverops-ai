"""Relational transactional persistence layer for workflow audit trails."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from app.models import (
    ExecutionResult,
    RecoveryAttempt,
    VerificationResult,
    Workflow,
    WorkflowResult,
    _current_timestamp,
)


def _safe_json_dumps(value: Any) -> str:
    """Safely serialize an object to a JSON string without risk of arbitrary execution."""
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return json.dumps({"raw_repr": repr(value)})


class StorageError(Exception):
    """Raised when an audit trail persistence error occurs."""
    pass


class Storage:
    """SQLite-backed relational audit store with strict parameterized SQL.

    Guarantees:
      - All queries use '?' parameter placeholders (zero f-string or '%' SQL formatting).
      - Strict foreign-key validation enabled via PRAGMA foreign_keys = ON.
      - Ephemeral databases clean up deterministically on Windows without lock retention.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._is_temp: bool = False
        self._conn: sqlite3.Connection | None = None

        if db_path is None or db_path == ":memory:":
            temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_file.close()
            self.db_path = Path(temp_file.name).resolve()
            self._is_temp = True
            atexit.register(self.close)
        else:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,
                    recovery_budget INTEGER NOT NULL,
                    total_recoveries_used INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    succeeded INTEGER NOT NULL DEFAULT 0,
                    escalated INTEGER NOT NULL DEFAULT 0,
                    escalation_reason TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    exception_raised INTEGER NOT NULL,
                    error_code TEXT NOT NULL,
                    error_message TEXT,
                    return_value_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    criterion_type TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    observed_value_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recovery_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    recovery_number INTEGER NOT NULL,
                    failure_category TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    applied INTEGER NOT NULL,
                    escalated INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id)
                );
                """
            )

    def create_workflow_run(self, workflow: Workflow) -> int:
        """Create parent workflow audit record and return its row ID."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO workflows (
                    workflow_name,
                    recovery_budget,
                    total_recoveries_used,
                    status,
                    succeeded,
                    escalated,
                    escalation_reason,
                    started_at,
                    completed_at
                ) VALUES (?, ?, 0, 'RUNNING', 0, 0, NULL, ?, NULL);
                """,
                (workflow.name, workflow.recovery_budget, _current_timestamp()),
            )
            return int(cursor.lastrowid)

    def complete_workflow_run(self, workflow_id: int, result: WorkflowResult) -> None:
        """Update workflow record with terminal status and completion timestamp."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                UPDATE workflows
                SET status = ?,
                    total_recoveries_used = ?,
                    succeeded = ?,
                    escalated = ?,
                    escalation_reason = ?,
                    completed_at = ?
                WHERE id = ?;
                """,
                (
                    result.status.value,
                    result.total_recoveries_used,
                    1 if result.succeeded else 0,
                    1 if result.escalated else 0,
                    result.escalation_reason,
                    _current_timestamp(),
                    workflow_id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(f"Workflow ID {workflow_id} not found for completion update")

    def record_execution(
        self,
        workflow_id: int,
        task_name: str,
        execution: ExecutionResult,
    ) -> None:
        """Persist a task execution attempt."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO task_executions (
                    workflow_id,
                    task_id,
                    task_name,
                    attempt_number,
                    action,
                    exception_raised,
                    error_code,
                    error_message,
                    return_value_json,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    workflow_id,
                    execution.task_id,
                    task_name,
                    execution.attempt_number,
                    execution.action.value,
                    1 if execution.exception_raised else 0,
                    execution.error_code.value,
                    execution.error_message,
                    _safe_json_dumps(execution.return_value),
                    execution.timestamp,
                ),
            )

    def record_verification(
        self,
        workflow_id: int,
        verification: VerificationResult,
    ) -> None:
        """Persist a verification outcome."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO verifications (
                    workflow_id,
                    task_id,
                    attempt_number,
                    criterion_type,
                    passed,
                    reason,
                    observed_value_json,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    workflow_id,
                    verification.task_id,
                    verification.attempt_number,
                    verification.criterion_type.value,
                    1 if verification.passed else 0,
                    verification.reason,
                    _safe_json_dumps(verification.observed_value),
                    verification.timestamp,
                ),
            )

    def record_recovery(
        self,
        workflow_id: int,
        recovery: RecoveryAttempt,
    ) -> None:
        """Persist a recovery attempt decision and remediation result."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO recovery_attempts (
                    workflow_id,
                    task_id,
                    recovery_number,
                    failure_category,
                    strategy,
                    applied,
                    escalated,
                    details,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    workflow_id,
                    recovery.task_id,
                    recovery.recovery_number,
                    recovery.failure_category.value,
                    recovery.strategy.value,
                    1 if recovery.applied else 0,
                    1 if recovery.escalated else 0,
                    recovery.details,
                    recovery.timestamp,
                ),
            )

    def get_workflow_runs(self) -> list[dict[str, Any]]:
        """Retrieve all workflow runs ordered by primary key."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM workflows ORDER BY id ASC;")
        return [dict(row) for row in cursor.fetchall()]

    def get_executions(self, workflow_id: int) -> list[dict[str, Any]]:
        """Retrieve all task executions for a given workflow run."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM task_executions WHERE workflow_id = ? ORDER BY id ASC;",
            (workflow_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_verifications(self, workflow_id: int) -> list[dict[str, Any]]:
        """Retrieve all verification records for a given workflow run."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM verifications WHERE workflow_id = ? ORDER BY id ASC;",
            (workflow_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_recoveries(self, workflow_id: int) -> list[dict[str, Any]]:
        """Retrieve all recovery records for a given workflow run."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM recovery_attempts WHERE workflow_id = ? ORDER BY id ASC;",
            (workflow_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close connection and unlink ephemeral database files if temporary."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._is_temp and self.db_path.exists():
            for suffix in ["", "-wal", "-shm", "-journal"]:
                p = self.db_path.with_name(self.db_path.name + suffix)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

