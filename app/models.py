"""Typed domain models used by the RecoverOps workflow engine.

The engine deliberately models a small, closed set of actions and recovery
strategies. This keeps the Phase 1 implementation deterministic and avoids
turning workflow input into executable code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from copy import deepcopy
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class TaskAction(str, Enum):
    """Safe, built-in actions understood by :class:`FailureSimulator`."""

    SUCCESS = "success"
    TRANSIENT_SERVICE = "transient_service"
    TIMEOUT = "timeout"
    MISSING_FILE = "missing_file"
    INVALID_DATA = "invalid_data"
    UNRECOVERABLE = "unrecoverable"


class FailureKind(str, Enum):
    """Failure categories that can be diagnosed by the recovery engine."""

    NONE = "none"
    TRANSIENT_SERVICE = "transient_service"
    TIMEOUT = "timeout"
    MISSING_FILE = "missing_file"
    INVALID_DATA = "invalid_data"
    VERIFICATION_FAILED = "verification_failed"
    UNRECOVERABLE = "unrecoverable"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Predefined safe recovery actions."""

    RETRY = "retry"
    RESET_SERVICE = "reset_service"
    RESTORE_FILE = "restore_file"
    SANITIZE_DATA = "sanitize_data"
    ESCALATE = "escalate"


class CriterionType(str, Enum):
    """Independent checks available to the verifier."""

    OUTPUT_EQUALS = "output_equals"
    OUTPUT_CONTAINS = "output_contains"
    OUTPUT_NOT_EMPTY = "output_not_empty"
    FILE_EXISTS = "file_exists"
    DATA_VALID = "data_valid"
    SERVICE_READY = "service_ready"


class VerificationStatus(str, Enum):
    """Result of an independent success-criteria check."""

    PASSED = "passed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Final state for one workflow task."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    ESCALATED = "escalated"


class WorkflowStatus(str, Enum):
    """Final state for a complete workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class SuccessCriteria:
    """A declarative condition that must independently pass.

    ``expected`` is required for equality, containment, and file-existence
    checks. The verifier, not the action executor, owns the decision that a
    task passed.
    """

    criterion: CriterionType
    expected: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, CriterionType):
            object.__setattr__(self, "criterion", CriterionType(self.criterion))
        required = {
            CriterionType.OUTPUT_EQUALS,
            CriterionType.OUTPUT_CONTAINS,
            CriterionType.FILE_EXISTS,
        }
        if self.criterion in required and self.expected is None:
            raise ValueError(f"{self.criterion.value} requires an expected value")
        object.__setattr__(self, "expected", deepcopy(self.expected))


@dataclass(frozen=True)
class Task:
    """A single safe operation and its explicit verification contract."""

    id: str
    name: str
    action: TaskAction
    success_criteria: SuccessCriteria
    parameters: Mapping[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    max_recoveries: int = 2
    idempotent: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("task id must not be empty")
        if not self.name or not self.name.strip():
            raise ValueError("task name must not be empty")
        if not isinstance(self.action, TaskAction):
            object.__setattr__(self, "action", TaskAction(self.action))
        if not isinstance(self.success_criteria, SuccessCriteria):
            raise TypeError("success_criteria must be a SuccessCriteria instance")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if type(self.max_recoveries) is not int or self.max_recoveries < 0:
            raise ValueError("max_recoveries must not be negative")
        # Deep-copy configuration so callers cannot change nested task
        # behavior during execution.
        object.__setattr__(self, "parameters", deepcopy(dict(self.parameters)))


@dataclass(frozen=True)
class Workflow:
    """An ordered collection of uniquely identified tasks."""

    name: str
    tasks: tuple[Task, ...]
    recovery_budget: int = 5
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("workflow name must not be empty")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if not self.tasks:
            raise ValueError("workflow must contain at least one task")
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("workflow task IDs must be unique")
        if type(self.recovery_budget) is not int or self.recovery_budget < 0:
            raise ValueError("recovery_budget must not be negative")


@dataclass(frozen=True)
class ExecutionResult:
    """The direct result of one action invocation.

    ``executor_succeeded`` only describes whether the action returned a usable
    result. It is intentionally not the final task status; verification must
    pass separately before the workflow can succeed.
    """

    workflow_id: str
    task_id: str
    attempt: int
    executor_succeeded: bool
    output: Any = None
    failure_kind: FailureKind = FailureKind.NONE
    error: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)

    @property
    def duration_ms(self) -> int:
        """Return elapsed execution time rounded down to milliseconds."""

        return max(0, int((self.finished_at - self.started_at).total_seconds() * 1000))


@dataclass(frozen=True)
class RecoveryAttempt:
    """An auditable application of a predefined recovery strategy."""

    workflow_id: str
    task_id: str
    attempt: int
    strategy: RecoveryStrategy
    reason: str
    succeeded: bool
    message: str
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class VerificationResult:
    """Independent evaluation of a task's explicit success criteria."""

    workflow_id: str
    task_id: str
    attempt: int
    status: VerificationStatus
    criterion: CriterionType
    evidence: Mapping[str, Any]
    error: str | None = None
    verified_at: datetime = field(default_factory=utc_now)

    @property
    def succeeded(self) -> bool:
        """Return whether the criterion passed."""

        return self.status is VerificationStatus.PASSED


@dataclass
class TaskOutcome:
    """Aggregated outcome and audit trail for one task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    executions: list[ExecutionResult] = field(default_factory=list)
    recoveries: list[RecoveryAttempt] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class WorkflowRunResult:
    """Complete in-memory report returned by the workflow executor."""

    workflow_id: str
    status: WorkflowStatus
    task_outcomes: dict[str, TaskOutcome]
    timeline: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Return true only when every task was independently verified."""

        return self.status is WorkflowStatus.SUCCEEDED
