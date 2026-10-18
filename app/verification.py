"""Independent verification of task outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .models import (
    CriterionType,
    ExecutionResult,
    Task,
    VerificationResult,
    VerificationStatus,
)


class Verifier:
    """Evaluate declarative criteria without trusting the executor's result."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None

    def verify(self, task: Task, execution: ExecutionResult) -> VerificationResult:
        """Return a pass/fail result for one execution attempt.

        An action that returns successfully can still fail here. In particular,
        the verifier never treats ``executor_succeeded`` as task success.
        """

        criterion = task.success_criteria
        if not execution.executor_succeeded:
            return self._result(
                execution,
                VerificationStatus.FAILED,
                criterion.criterion,
                {"executor_succeeded": False, "error": execution.error},
                execution.error or "action did not produce a result",
            )

        try:
            passed, evidence, error = self._evaluate(criterion.criterion, criterion.expected, execution.output)
        except (OSError, TypeError, ValueError) as exc:
            passed, evidence, error = False, {"exception": str(exc)}, f"verification error: {exc}"

        return self._result(
            execution,
            VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            criterion.criterion,
            evidence,
            error,
        )

    def _evaluate(
        self, criterion: CriterionType, expected: Any, output: Any
    ) -> tuple[bool, dict[str, Any], str | None]:
        if criterion is CriterionType.OUTPUT_EQUALS:
            passed = output == expected
            return passed, {"observed": output, "expected": expected}, None if passed else "output did not equal expected value"

        if criterion is CriterionType.OUTPUT_CONTAINS:
            passed = self._contains(output, expected)
            return passed, {"observed": output, "expected": expected}, None if passed else "output did not contain expected value"

        if criterion is CriterionType.OUTPUT_NOT_EMPTY:
            # ``bool`` covers all built-in empty containers (including tuples,
            # sets, bytes, and bytearrays) while rejecting None and falsey
            # scalar values as non-meaningful output.
            passed = bool(output)
            return passed, {"observed": output}, None if passed else "output was empty"

        if criterion is CriterionType.FILE_EXISTS:
            path = self._safe_path(expected)
            passed = path.is_file()
            return passed, {"path": str(path), "exists": passed}, None if passed else "required file does not exist"

        if criterion is CriterionType.DATA_VALID:
            passed = isinstance(output, Mapping) and output.get("valid") is True
            return passed, {"observed": output, "valid": passed}, None if passed else "data payload failed the validity check"

        if criterion is CriterionType.SERVICE_READY:
            passed = (
                isinstance(output, Mapping) and output.get("service_ready") is True
            ) or output == "ready"
            return passed, {"observed": output, "service_ready": passed}, None if passed else "service is not ready"

        return False, {"observed": output}, f"unsupported criterion: {criterion}"

    def _safe_path(self, value: object) -> Path:
        candidate = Path(str(value))
        if self.base_dir is None:
            return candidate.resolve()
        resolved = (self.base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            resolved.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError("verification path is outside the verifier root") from exc
        return resolved

    @staticmethod
    def _contains(output: Any, expected: Any) -> bool:
        if isinstance(output, Mapping) and isinstance(expected, Mapping):
            return all(key in output and output[key] == value for key, value in expected.items())
        try:
            return expected in output
        except TypeError:
            return False

    @staticmethod
    def _result(
        execution: ExecutionResult,
        status: VerificationStatus,
        criterion: CriterionType,
        evidence: dict[str, Any],
        error: str | None,
    ) -> VerificationResult:
        return VerificationResult(
            workflow_id=execution.workflow_id,
            task_id=execution.task_id,
            attempt=execution.attempt,
            status=status,
            criterion=criterion,
            evidence=evidence,
            error=error,
        )
