"""Domain models, enumeration types, and validation contracts for RecoverOps AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _current_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class TaskAction(Enum):
    """Supported deterministic task actions."""
    SUCCESS = "success"
    TRANSIENT_SERVICE = "transient_service"
    TIMEOUT = "timeout"
    MISSING_FILE = "missing_file"
    INVALID_DATA = "invalid_data"
    UNRECOVERABLE = "unrecoverable"


class CriterionType(Enum):
    """Supported declarative verification criteria."""
    OUTPUT_EQUALS = "output_equals"
    OUTPUT_CONTAINS = "output_contains"
    NON_EMPTY_OUTPUT = "non_empty_output"
    FILE_EXISTS = "file_exists"
    DATA_VALID = "data_valid"
    SERVICE_READY = "service_ready"


class FailureCategory(Enum):
    """Taxonomy of failure classifications."""
    NONE = "none"
    TRANSIENT_SERVICE_ERROR = "transient_service_error"
    TIMEOUT_ERROR = "timeout_error"
    MISSING_FILE_ERROR = "missing_file_error"
    INVALID_DATA_ERROR = "invalid_data_error"
    VERIFICATION_FAILURE = "verification_failure"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class RecoveryStrategy(Enum):
    """Allowed bounded recovery strategies."""
    RETRY = "retry"
    RESET_SERVICE = "reset_service"
    RESTORE_FILE = "restore_file"
    SANITIZE_DATA = "sanitize_data"
    ESCALATE = "escalate"


class TaskStatus(Enum):
    """Execution status for a task within a workflow run."""
    SUCCEEDED = "succeeded"
    RECOVERED = "recovered"
    ESCALATED = "escalated"
    FAILED = "failed"


class WorkflowStatus(Enum):
    """Aggregate lifecycle state of an executed workflow."""
    COMPLETED = "completed"
    COMPLETED_WITH_RECOVERY = "completed_with_recovery"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass(frozen=True)
class SuccessCriteria:
    """Declarative specification for verifying task execution correctness."""
    criterion_type: CriterionType
    expected: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.criterion_type, CriterionType):
            raise TypeError(
                f"criterion_type must be an instance of CriterionType, got {type(self.criterion_type).__name__}"
            )

        if self.criterion_type == CriterionType.FILE_EXISTS:
            if not isinstance(self.expected, str) or not self.expected.strip():
                raise ValueError("FILE_EXISTS criterion requires a non-empty relative file path string as 'expected'")

        elif self.criterion_type == CriterionType.OUTPUT_CONTAINS:
            if self.expected is None:
                raise ValueError("OUTPUT_CONTAINS criterion requires a non-None expected value")

        elif self.criterion_type == CriterionType.DATA_VALID:
            if self.expected is None:
                raise ValueError("DATA_VALID criterion requires a meaningful schema or required-key definition")
            if isinstance(self.expected, (list, tuple, set)) and len(self.expected) == 0:
                raise ValueError("DATA_VALID criterion requires at least one required key in expected sequence")
            if isinstance(self.expected, dict) and len(self.expected) == 0:
                raise ValueError("DATA_VALID criterion schema dictionary cannot be empty")


@dataclass(frozen=True)
class Task:
    """A single deterministic task unit in an automated workflow."""
    id: str
    name: str
    action: TaskAction
    parameters: dict[str, Any] = field(default_factory=dict)
    success_criteria: SuccessCriteria = field(
        default_factory=lambda: SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT)
    )
    idempotent: bool = True
    max_attempts: int = 3
    max_recoveries: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Task id cannot be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Task name cannot be empty")
        if not isinstance(self.action, TaskAction):
            raise TypeError(
                f"Task action must be an instance of TaskAction, got {type(self.action).__name__}"
            )
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"Task parameters must be a dictionary, got {type(self.parameters).__name__}"
            )
        if not isinstance(self.success_criteria, SuccessCriteria):
            raise TypeError(
                f"success_criteria must be an instance of SuccessCriteria, got {type(self.success_criteria).__name__}"
            )
        if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError(f"max_attempts must be an integer >= 1, got {self.max_attempts}")
        if not isinstance(self.max_recoveries, int) or self.max_recoveries < 0:
            raise ValueError(f"max_recoveries must be an integer >= 0, got {self.max_recoveries}")


@dataclass(frozen=True)
class Workflow:
    """An ordered sequence of tasks governed by a shared recovery budget."""
    name: str
    tasks: tuple[Task, ...]
    recovery_budget: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Workflow name cannot be empty")

        if isinstance(self.tasks, list):
            object.__setattr__(self, "tasks", tuple(self.tasks))
        elif not isinstance(self.tasks, tuple):
            raise TypeError("tasks must be a sequence of Task instances")

        if len(self.tasks) == 0:
            raise ValueError("Workflow must contain at least one task")

        seen_ids: set[str] = set()
        for t in self.tasks:
            if not isinstance(t, Task):
                raise TypeError(f"All items in tasks must be Task instances, found {type(t).__name__}")
            if t.id in seen_ids:
                raise ValueError(f"Duplicate task id '{t.id}' found in workflow")
            seen_ids.add(t.id)

        if not isinstance(self.recovery_budget, int) or self.recovery_budget < 0:
            raise ValueError(f"recovery_budget must be >= 0, got {self.recovery_budget}")


@dataclass(frozen=True)
class ExecutionResult:
    """Raw outcome from executing a single task attempt."""
    task_id: str
    attempt_number: int
    action: TaskAction
    return_value: Any
    exception_raised: bool
    error_code: FailureCategory
    error_message: str | None
    timestamp: str = field(default_factory=_current_timestamp)


@dataclass(frozen=True)
class VerificationResult:
    """Result of evaluating declarative success criteria against an execution."""
    task_id: str
    attempt_number: int
    criterion_type: CriterionType
    passed: bool
    reason: str
    observed_value: Any = None
    timestamp: str = field(default_factory=_current_timestamp)


@dataclass(frozen=True)
class RecoveryAttempt:
    """Record of an automated recovery strategy evaluation and application."""
    task_id: str
    recovery_number: int
    failure_category: FailureCategory
    strategy: RecoveryStrategy
    applied: bool
    details: str
    escalated: bool
    timestamp: str = field(default_factory=_current_timestamp)


@dataclass(frozen=True)
class TaskOutcome:
    """Comprehensive summary of task lifecycle across all attempts and recoveries."""
    task_id: str
    task_name: str
    status: TaskStatus
    attempts_used: int
    recoveries_used: int
    final_output: Any
    escalated: bool
    escalation_reason: str | None
    executions: tuple[ExecutionResult, ...] = field(default_factory=tuple)
    verifications: tuple[VerificationResult, ...] = field(default_factory=tuple)
    recoveries: tuple[RecoveryAttempt, ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        """Return True only if task reached SUCCEEDED or RECOVERED status."""
        return self.status in (TaskStatus.SUCCEEDED, TaskStatus.RECOVERED)


@dataclass(frozen=True)
class WorkflowResult:
    """Terminal outcome of an orchestrated workflow execution."""
    workflow_name: str
    status: WorkflowStatus
    succeeded: bool
    escalated: bool
    escalation_reason: str | None
    total_recoveries_used: int
    recovery_budget: int
    task_outcomes: tuple[TaskOutcome, ...]
    timeline: tuple[str, ...]

