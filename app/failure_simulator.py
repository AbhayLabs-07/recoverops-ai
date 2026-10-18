"""Deterministic, safe task actions for demonstrations and tests.

The simulator is intentionally not a shell runner. It exposes a closed set of
operations and keeps all recovery mutations inside those operations, making it
useful for reproducible Phase 1 behavior without executing untrusted input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    ExecutionResult,
    FailureKind,
    RecoveryStrategy,
    Task,
    TaskAction,
    utc_now,
)


class FailureSimulator:
    """Execute deterministic built-in actions and apply safe test recoveries."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else None
        if self.root_dir is not None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        self._state: dict[tuple[str, str], dict[str, bool]] = {}

    def execute(self, task: Task, workflow_id: str, attempt: int) -> ExecutionResult:
        """Run one known action and return a structured result.

        Failure timing is based on the attempt number and task parameters, so
        the same task always reproduces the same scenario.
        """

        started = utc_now()
        state = self._state.setdefault((workflow_id, task.id), {})
        try:
            if task.action is TaskAction.SUCCESS:
                result = self._success(task, workflow_id, attempt, started)
            elif task.action is TaskAction.TRANSIENT_SERVICE:
                result = self._transient_service(task, workflow_id, attempt, started, state)
            elif task.action is TaskAction.TIMEOUT:
                result = self._timeout(task, workflow_id, attempt, started, state)
            elif task.action is TaskAction.MISSING_FILE:
                result = self._missing_file(task, workflow_id, attempt, started)
            elif task.action is TaskAction.INVALID_DATA:
                result = self._invalid_data(task, workflow_id, attempt, started, state)
            else:
                result = self._failure(
                    workflow_id,
                    task,
                    attempt,
                    started,
                    FailureKind.UNRECOVERABLE,
                    "the simulated error is not recoverable automatically",
                )
        except (OSError, ValueError, TypeError) as exc:
            result = self._failure(
                workflow_id,
                task,
                attempt,
                started,
                FailureKind.UNKNOWN,
                f"safe simulator operation failed: {exc}",
            )
        return result

    def apply_recovery(
        self, task: Task, strategy: RecoveryStrategy, workflow_id: str = ""
    ) -> tuple[bool, str]:
        """Apply one predefined recovery mutation for ``task``.

        No strategy accepts a command string or dynamically generated code.
        ``False`` means the known recovery could not be safely applied.
        """

        state = self._state.setdefault((workflow_id, task.id), {})
        if bool(task.parameters.get("never_recover", False)):
            if strategy is RecoveryStrategy.ESCALATE:
                return False, "the deterministic scenario is configured to remain failed"
            # The recovery decision itself is safe and auditable, but this
            # fixture intentionally keeps returning the same failure so the
            # executor's retry and budget limits can be exercised.
            return True, "recovery recorded; deterministic scenario remains failed"

        if strategy is RecoveryStrategy.RESET_SERVICE:
            state["service_reset"] = True
            return True, "service state reset; a subsequent attempt may proceed"
        if strategy is RecoveryStrategy.RETRY:
            state["timeout_recovered"] = True
            return True, "retry approved for the known transient condition"
        if strategy is RecoveryStrategy.RESTORE_FILE:
            path_value = task.parameters.get("path")
            if path_value is None:
                return False, "missing-file recovery requires a path"
            try:
                path = self._safe_path(path_value)
                content = str(task.parameters.get("recovery_content", "restored"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            except (OSError, ValueError) as exc:
                return False, f"file could not be restored safely: {exc}"
            state["file_restored"] = True
            return True, f"restored the known file target {path}"
        if strategy is RecoveryStrategy.SANITIZE_DATA:
            state["data_sanitized"] = True
            return True, "sanitized the known simulated data payload"
        return False, "human intervention is required for this failure"

    def _success(
        self, task: Task, workflow_id: str, attempt: int, started: Any
    ) -> ExecutionResult:
        output = task.parameters.get("output", {"ok": True})
        return self._success_result(workflow_id, task, attempt, started, output)

    def _transient_service(
        self,
        task: Task,
        workflow_id: str,
        attempt: int,
        started: Any,
        state: dict[str, bool],
    ) -> ExecutionResult:
        failures = int(task.parameters.get("failures_before_success", 1))
        should_fail = bool(task.parameters.get("never_recover", False)) or (
            not state.get("service_reset", False) and attempt <= failures
        )
        if should_fail:
            return self._failure(
                workflow_id,
                task,
                attempt,
                started,
                FailureKind.TRANSIENT_SERVICE,
                "simulated transient service unavailability",
            )
        return self._success_result(
            workflow_id,
            task,
            attempt,
            started,
            task.parameters.get("output", {"service_ready": True}),
        )

    def _timeout(
        self,
        task: Task,
        workflow_id: str,
        attempt: int,
        started: Any,
        state: dict[str, bool],
    ) -> ExecutionResult:
        failures = int(task.parameters.get("failures_before_success", 1))
        should_fail = bool(task.parameters.get("never_recover", False)) or (
            not state.get("timeout_recovered", False) and attempt <= failures
        )
        if should_fail:
            return self._failure(
                workflow_id,
                task,
                attempt,
                started,
                FailureKind.TIMEOUT,
                "simulated operation timeout",
            )
        return self._success_result(
            workflow_id,
            task,
            attempt,
            started,
            task.parameters.get("output", {"completed": True}),
        )

    def _missing_file(
        self, task: Task, workflow_id: str, attempt: int, started: Any
    ) -> ExecutionResult:
        path_value = task.parameters.get("path")
        if path_value is None:
            return self._failure(
                workflow_id,
                task,
                attempt,
                started,
                FailureKind.MISSING_FILE,
                "the task did not define a file path",
            )
        path = self._safe_path(path_value)
        if not path.exists():
            return self._failure(
                workflow_id,
                task,
                attempt,
                started,
                FailureKind.MISSING_FILE,
                f"required file does not exist: {path}",
            )
        return self._success_result(
            workflow_id,
            task,
            attempt,
            started,
            task.parameters.get(
                "output", {"path": str(path), "exists": True, "content": path.read_text(encoding="utf-8")}
            ),
        )

    def _invalid_data(
        self,
        task: Task,
        workflow_id: str,
        attempt: int,
        started: Any,
        state: dict[str, bool],
    ) -> ExecutionResult:
        if state.get("data_sanitized", False):
            output = task.parameters.get(
                "valid_output", {"valid": True, "value": task.parameters.get("value", "sanitized")}
            )
        else:
            output = task.parameters.get(
                "invalid_output", {"valid": False, "error": "simulated invalid data"}
            )
        # The action returns normally, but the verifier must still reject the
        # invalid payload before any task is considered successful.
        return self._success_result(workflow_id, task, attempt, started, output)

    def _safe_path(self, value: object) -> Path:
        """Resolve a file target and prevent root-directory traversal."""

        candidate = Path(str(value))
        if self.root_dir is None:
            return candidate.resolve()
        resolved = (self.root_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as exc:
            raise ValueError("file target is outside the simulator root") from exc
        return resolved

    @staticmethod
    def _success_result(
        workflow_id: str,
        task: Task,
        attempt: int,
        started: Any,
        output: Any,
    ) -> ExecutionResult:
        return ExecutionResult(
            workflow_id=workflow_id,
            task_id=task.id,
            attempt=attempt,
            executor_succeeded=True,
            output=output,
            started_at=started,
            finished_at=utc_now(),
        )

    @staticmethod
    def _failure(
        workflow_id: str,
        task: Task,
        attempt: int,
        started: Any,
        kind: FailureKind,
        error: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            workflow_id=workflow_id,
            task_id=task.id,
            attempt=attempt,
            executor_succeeded=False,
            failure_kind=kind,
            error=error,
            started_at=started,
            finished_at=utc_now(),
        )
