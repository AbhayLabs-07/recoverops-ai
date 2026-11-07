"""Deterministic simulated execution environment and fault injection sandbox."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from app.models import (
    ExecutionResult,
    FailureCategory,
    Task,
    TaskAction,
)


class SimulatorError(Exception):
    """Base exception for deterministic simulator faults."""
    pass


class SafePathViolationError(ValueError, SimulatorError):
    """Raised when an operation attempts to escape the simulator sandbox."""
    pass


class UnrestorableFileError(SimulatorError):
    """Raised when a file restoration cannot be fulfilled."""
    pass


class FailureSimulator:
    """Deterministic environment simulator for task execution and fault injection.

    Guarantees:
      - All filesystem mutations are confined to a designated or temporary sandbox directory.
      - Path resolution strictly prevents directory traversal or absolute path escapes.
      - Failures and state remediations are completely deterministic and isolated per task ID.
    """

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self._is_managed_temp: bool = False
        if root_dir is None:
            self.root_dir = Path(tempfile.mkdtemp(prefix="recoverops_sim_")).resolve()
            self._is_managed_temp = True
            atexit.register(self.cleanup)
        else:
            self.root_dir = Path(root_dir).resolve()
            self.root_dir.mkdir(parents=True, exist_ok=True)

        self._service_ready: dict[str, bool] = {}
        self._sanitized_data: dict[str, dict[str, Any]] = {}
        self._restored_files: set[str] = set()
        self._logical_backoffs: dict[str, int] = {}
        self._deadlocks_resolved: set[str] = set()
        self._sandbox_temp_files: set[str] = set()
        self._quota_resolved: set[str] = set()
        self._flaky_service_resets: dict[str, int] = {}
        self._partial_corruptions_sanitized: set[str] = set()
        self._completed_task_ids: set[str] = set()

    def resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a path strictly inside the sandbox root.

        Rejects:
          - Non-string input
          - Empty or whitespace strings
          - Absolute paths (POSIX /root or Windows C:\\)
          - Windows drive specifications (e.g., C:file.txt)
          - Relative parent traversals (..) that resolve outside root
        """
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("Path traversal outside simulator root is forbidden")

        clean_path_str = relative_path.strip()

        # Reject Windows drive letters (e.g. C:, D:)
        drive, _ = os.path.splitdrive(clean_path_str)
        if drive:
            raise ValueError("Path traversal outside simulator root is forbidden")

        p = Path(clean_path_str)
        if p.is_absolute():
            raise ValueError("Path traversal outside simulator root is forbidden")

        # Explicitly check for parent directory traversal tokens
        for part in p.parts:
            if part == "..":
                raise ValueError("Path traversal outside simulator root is forbidden")

        resolved_root = self.root_dir.resolve()
        candidate = (resolved_root / p).resolve()

        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            raise ValueError("Path traversal outside simulator root is forbidden")

        return candidate

    def execute(self, task: Task, attempt_number: int) -> ExecutionResult:
        """Execute a task deterministically based on its action and attempt state."""
        action = task.action
        params = task.parameters

        if action == TaskAction.SUCCESS:
            if "create_file" in params:
                file_rel = params["create_file"]
                file_p = self.resolve_safe_path(file_rel)
                file_p.parent.mkdir(parents=True, exist_ok=True)
                content = params.get("file_content", "simulated deterministic output content")
                file_p.write_text(content, encoding="utf-8")

            if "service_ready" in params:
                self._service_ready[task.id] = bool(params["service_ready"])

            output = params.get("output", {"status": "ok"})
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=output,
                exception_raised=False,
                error_code=FailureCategory.NONE,
                error_message=None,
            )

        if action == TaskAction.TRANSIENT_SERVICE:
            succeed_on_attempt = params.get("succeed_on_attempt", 2)
            has_been_reset = self._service_ready.get(task.id, False)

            if has_been_reset or attempt_number >= succeed_on_attempt:
                self._service_ready[task.id] = True
                output = params.get("output", {"service": "online", "healthy": True})
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            self._service_ready[task.id] = False
            msg = params.get(
                "error_message",
                f"Transient service communication failure on attempt {attempt_number}",
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.TRANSIENT_SERVICE_ERROR,
                error_message=msg,
            )

        if action == TaskAction.TIMEOUT:
            succeed_on_attempt = params.get("succeed_on_attempt", 2)
            if attempt_number >= succeed_on_attempt:
                output = params.get("output", {"completed": True, "timed_out": False})
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get("error_message", "Simulated deterministic operation timeout")
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.TIMEOUT_ERROR,
                error_message=msg,
            )

        if action == TaskAction.MISSING_FILE:
            file_rel = params.get("file_path", "required_input.json")
            target_path = self.resolve_safe_path(file_rel)

            if target_path.is_file():
                raw_text = target_path.read_text(encoding="utf-8")
                try:
                    loaded = json.loads(raw_text)
                except Exception:
                    loaded = {"path": file_rel, "content": raw_text}
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=loaded,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = f"Required file missing: {file_rel}"
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.MISSING_FILE_ERROR,
                error_message=msg,
            )

        if action == TaskAction.INVALID_DATA:
            if task.id in self._sanitized_data:
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=self._sanitized_data[task.id],
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            raw_data = params.get("data")
            required_keys = params.get("required_keys", ["id", "valid"])

            is_valid = isinstance(raw_data, dict)
            if is_valid:
                for k in required_keys:
                    if k not in raw_data or raw_data[k] is None:
                        is_valid = False
                        break
                if raw_data.get("valid") is False:
                    is_valid = False

            if is_valid:
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=raw_data,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get("error_message", "Data validation failed: missing or invalid keys")
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=raw_data,
                exception_raised=True,
                error_code=FailureCategory.INVALID_DATA_ERROR,
                error_message=msg,
            )

        if action == TaskAction.UNRECOVERABLE:
            msg = params.get("error_message", "Unrecoverable hardware/permission fault")
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.UNRECOVERABLE_ERROR,
                error_message=msg,
            )

        if action == TaskAction.RATE_LIMITED:
            succeed_on_attempt = params.get("succeed_on_attempt", 2)
            backoff_step = self._logical_backoffs.get(task.id, 0) + 1
            self._logical_backoffs[task.id] = backoff_step

            if attempt_number >= succeed_on_attempt:
                output = params.get(
                    "output",
                    {"rate_limit_cleared": True, "logical_backoff_step": backoff_step},
                )
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                f"Rate limit exceeded (HTTP 429). Logical backoff step: {backoff_step}",
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.RATE_LIMIT_EXCEEDED,
                error_message=msg,
            )

        if action == TaskAction.DEADLOCK:
            if task.id in self._deadlocks_resolved:
                output = params.get(
                    "output",
                    {"transaction": "committed", "replayed": True},
                )
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                "Transaction deadlock detected on SQLite table lock; rollback required",
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.TRANSACTION_DEADLOCK,
                error_message=msg,
            )

        if action == TaskAction.DISK_QUOTA:
            if task.id in self._quota_resolved:
                output = params.get(
                    "output",
                    {"disk_space_available": True, "quota_cleared": True},
                )
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                "Disk quota exceeded: temporary workspace storage full (99.8% capacity)",
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.DISK_QUOTA_EXCEEDED,
                error_message=msg,
            )

        if action == TaskAction.FLAKY_SERVICE:
            # Flaky service: under high hidden load, RESET_SERVICE alone does not cure it.
            # Only when reset has been tried AND backoff retry is applied on attempt >= succeed_on_attempt
            # or when reset count reaches required resets.
            resets_done = self._flaky_service_resets.get(task.id, 0)
            required_resets = params.get("required_resets", 1)
            min_attempt = params.get("succeed_on_attempt", 2)
            if resets_done >= required_resets and attempt_number >= min_attempt:
                output = params.get("output", {"service_status": "healthy", "flaky_load_cleared": True})
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                f"Flaky service degraded under load (resets={resets_done}/{required_resets}, attempt={attempt_number})"
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.FLAKY_SERVICE_DEGRADED,
                error_message=msg,
            )

        if action == TaskAction.PARTIAL_CORRUPTION:
            # Partial corruption: only specific sub-fields corrupted; sanitize works if applied
            if task.id in self._partial_corruptions_sanitized or task.id in self._sanitized_data:
                clean_data = self._sanitized_data.get(task.id, params.get("clean_data", {"id": task.id, "valid": True}))
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=clean_data,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                "Partial payload corruption: corrupted checksum on metadata header segment"
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.PARTIALLY_CORRUPTED_DATA,
                error_message=msg,
            )

        if action == TaskAction.CASCADING_FAILURE:
            # Cascading failure: fails if an upstream task failed in recent execution
            upstream_task = params.get("upstream_task_id")
            upstream_failed = params.get("upstream_failed", True)
            if upstream_task and upstream_task in self._completed_task_ids:
                upstream_failed = False

            if not upstream_failed or attempt_number >= params.get("succeed_on_attempt", 99):
                output = params.get("output", {"cascading_resolved": True})
                return ExecutionResult(
                    task_id=task.id,
                    attempt_number=attempt_number,
                    action=action,
                    return_value=output,
                    exception_raised=False,
                    error_code=FailureCategory.NONE,
                    error_message=None,
                )

            msg = params.get(
                "error_message",
                f"Cascading upstream dependency failure from '{upstream_task or 'parent'}'; downstream circuit broken"
            )
            return ExecutionResult(
                task_id=task.id,
                attempt_number=attempt_number,
                action=action,
                return_value=None,
                exception_raised=True,
                error_code=FailureCategory.CASCADED_UPSTREAM_FAILURE,
                error_message=msg,
            )

        raise ValueError(f"Unsupported action: {action}")

    def reset_service(self, task: Task) -> str:
        """Reset service health state for a task."""
        self._service_ready[task.id] = True
        self._flaky_service_resets[task.id] = self._flaky_service_resets.get(task.id, 0) + 1
        return f"Service state reset to online for task '{task.id}'"

    def restore_file(self, task: Task) -> str:
        """Restore missing file from configured backup content."""
        params = task.parameters
        if params.get("unrestorable") is True:
            raise UnrestorableFileError(f"File for task '{task.id}' cannot be restored")

        file_rel = params.get("file_path", "required_input.json")
        target_path = self.resolve_safe_path(file_rel)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        backup_content = params.get("restore_content")
        if backup_content is None:
            backup_content = json.dumps({"id": task.id, "restored": True})
        elif isinstance(backup_content, dict):
            backup_content = json.dumps(backup_content)
        elif not isinstance(backup_content, str):
            backup_content = str(backup_content)

        target_path.write_text(backup_content, encoding="utf-8")
        self._restored_files.add(file_rel)
        return f"File '{file_rel}' successfully restored for task '{task.id}'"

    def sanitize_data(self, task: Task) -> str:
        """Sanitize payload into clean deterministic data."""
        params = task.parameters
        if "sanitized_data" in params and isinstance(params["sanitized_data"], dict):
            clean = dict(params["sanitized_data"])
        else:
            raw = params.get("data")
            if isinstance(raw, dict):
                clean = {k: v for k, v in raw.items() if v is not None}
            else:
                clean = {}
            clean["id"] = clean.get("id", task.id)
            clean["valid"] = True

        self._sanitized_data[task.id] = clean
        self._partial_corruptions_sanitized.add(task.id)
        return f"Payload sanitized with keys {sorted(clean.keys())} for task '{task.id}'"

    def register_sandbox_temp_file(self, relative_path: str, content: str = "temp") -> Path:
        """Register and create a temporary artifact strictly confined to the sandbox."""
        p = self.resolve_safe_path(relative_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self._sandbox_temp_files.add(relative_path)
        return p

    def rollback_replay(self, task: Task) -> str:
        """Rollback conflicting transaction and allow deterministic replayed execution."""
        self._deadlocks_resolved.add(task.id)
        return f"Transaction deadlock resolved: rolled back and prepared for replay on task '{task.id}'"

    def purge_temp(self, task: Task) -> str:
        """Purge ONLY sandbox-registered temporary artifacts, strictly within the sandbox root."""
        purged_count = 0
        for rel in list(self._sandbox_temp_files):
            try:
                target_p = self.resolve_safe_path(rel)
                if target_p.is_file():
                    target_p.unlink()
                    purged_count += 1
            except Exception:
                pass
            self._sandbox_temp_files.discard(rel)

        self._quota_resolved.add(task.id)
        return f"Purged {purged_count} sandbox-registered temporary artifacts; disk quota restored for task '{task.id}'"

    def is_service_ready(self, task_id: str) -> bool:
        """Query if service is currently online for given task ID."""
        return self._service_ready.get(task_id, False)

    def cleanup(self) -> None:
        """Safely clean up managed temporary sandbox directory."""
        if self._is_managed_temp and self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)

    def __enter__(self) -> FailureSimulator:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

