"""Independent declarative verification engine for execution outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    Task,
    VerificationResult,
)


class Verifier:
    """Evaluates declarative success criteria independently of execution status."""

    def verify(
        self,
        task: Task,
        execution_result: ExecutionResult,
        simulator: FailureSimulator,
    ) -> VerificationResult:
        """Independently evaluate task success criteria.

        CRITICAL INVARIANT:
        A return without an unhandled exception does not equate to domain success.
        If an execution raised an exception, verification immediately returns False
        with an explicit reason, preventing unverified success.
        """
        crit = task.success_criteria
        attempt = execution_result.attempt_number
        val = execution_result.return_value

        if execution_result.exception_raised:
            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=False,
                reason="Execution failed with exception prior to verification",
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.OUTPUT_EQUALS:
            matches = val == crit.expected
            reason = (
                f"Observed value matches expected '{crit.expected}'"
                if matches
                else f"Observed value '{val}' does not equal expected '{crit.expected}'"
            )
            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=matches,
                reason=reason,
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.OUTPUT_CONTAINS:
            passed = False
            reason = ""
            exp = crit.expected

            if isinstance(exp, dict):
                if isinstance(val, dict):
                    missing_or_mismatch = [
                        k for k, v in exp.items() if k not in val or val[k] != v
                    ]
                    if not missing_or_mismatch:
                        passed = True
                        reason = "All expected key-value pairs present in output dictionary"
                    else:
                        passed = False
                        reason = f"Mismatch in expected dictionary keys: {missing_or_mismatch}"
                else:
                    passed = False
                    reason = f"Expected dict subset but observed type is {type(val).__name__}"
            elif isinstance(exp, str):
                if isinstance(val, str) and exp in val:
                    passed = True
                    reason = f"Substring '{exp}' found in output text"
                elif isinstance(val, (list, tuple, set)) and exp in val:
                    passed = True
                    reason = f"Element '{exp}' present in output collection"
                else:
                    passed = False
                    reason = f"Expected substring or element '{exp}' not found in output"
            elif isinstance(exp, (list, tuple, set)):
                if isinstance(val, (list, tuple, set)):
                    all_found = all(item in val for item in exp)
                    if all_found:
                        passed = True
                        reason = "All expected items present in output collection"
                    else:
                        passed = False
                        reason = "One or more expected items missing from output collection"
                else:
                    passed = False
                    reason = f"Expected collection subset but observed type is {type(val).__name__}"
            else:
                try:
                    if exp in val:
                        passed = True
                        reason = f"Element '{exp}' is contained within output"
                    else:
                        passed = False
                        reason = f"Element '{exp}' is not contained within output"
                except TypeError:
                    passed = False
                    reason = f"Cannot evaluate containment of '{exp}' in type {type(val).__name__}"

            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=passed,
                reason=reason,
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.NON_EMPTY_OUTPUT:
            if val is None:
                passed = False
                reason = "Observed value is None"
            elif hasattr(val, "__len__"):
                passed = len(val) > 0
                reason = (
                    f"Collection/string length is {len(val)}"
                    if passed
                    else "Observed collection or string is empty"
                )
            else:
                passed = bool(val)
                reason = "Observed truthy value" if passed else "Observed falsy value"

            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=passed,
                reason=reason,
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.FILE_EXISTS:
            file_rel = str(crit.expected)
            try:
                target_path: Path = simulator.resolve_safe_path(file_rel)
                exists = target_path.exists()
                is_file = target_path.is_file()
                size = target_path.stat().st_size if exists else 0
                non_empty = size > 0

                if exists and is_file and non_empty:
                    passed = True
                    reason = f"File '{file_rel}' exists, is a regular file, and contains {size} bytes"
                elif exists and is_file and not non_empty:
                    passed = False
                    reason = f"File '{file_rel}' exists but is empty (0 bytes)"
                elif exists and not is_file:
                    passed = False
                    reason = f"Path '{file_rel}' exists but is not a regular file"
                else:
                    passed = False
                    reason = f"File '{file_rel}' does not exist in sandbox"
            except Exception as e:
                passed = False
                reason = f"File verification encountered path resolution error: {e}"

            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=passed,
                reason=reason,
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.DATA_VALID:
            if not isinstance(val, dict):
                return VerificationResult(
                    task_id=task.id,
                    attempt_number=attempt,
                    criterion_type=crit.criterion_type,
                    passed=False,
                    reason=f"Observed value is not a dictionary (type={type(val).__name__})",
                    observed_value=val,
                )

            exp = crit.expected
            if isinstance(exp, dict):
                required_keys = list(exp.keys())
            elif isinstance(exp, (list, tuple, set)):
                required_keys = list(exp)
            else:
                required_keys = ["id", "valid"]

            missing = [k for k in required_keys if k not in val or val[k] is None]
            valid_flag = val.get("valid", True) is not False

            if missing:
                passed = False
                reason = f"Missing or null required keys in data payload: {missing}"
            elif not valid_flag:
                passed = False
                reason = "Payload explicitly specifies invalid status ('valid': False)"
            else:
                passed = True
                reason = f"Payload contains all valid required keys {required_keys}"

            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=passed,
                reason=reason,
                observed_value=val,
            )

        if crit.criterion_type == CriterionType.SERVICE_READY:
            sim_ready = simulator.is_service_ready(task.id)
            output_ready = isinstance(val, dict) and val.get("healthy") is True

            if sim_ready and output_ready:
                passed = True
                reason = f"Service for task '{task.id}' independently verified online and healthy"
            elif not sim_ready:
                passed = False
                reason = f"Service simulator state indicates service for '{task.id}' is not ready"
            else:
                passed = False
                reason = "Simulator reports service online, but returned output healthy flag is False"

            return VerificationResult(
                task_id=task.id,
                attempt_number=attempt,
                criterion_type=crit.criterion_type,
                passed=passed,
                reason=reason,
                observed_value=val,
            )

        raise ValueError(f"Unsupported criterion type: {crit.criterion_type}")

