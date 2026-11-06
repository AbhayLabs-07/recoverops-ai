"""Unit tests for domain models, validation contracts, and enum types."""

from __future__ import annotations

import unittest

from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    TaskOutcome,
    TaskStatus,
    VerificationResult,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
)


class TestModels(unittest.TestCase):
    """Test model validation rules, enum specifications, and invariant enforcement."""

    def test_enum_exact_names_and_values(self) -> None:
        """Verify exact enum names and lowercase string values."""
        self.assertEqual(TaskAction.SUCCESS.value, "success")
        self.assertEqual(TaskAction.TRANSIENT_SERVICE.value, "transient_service")
        self.assertEqual(TaskAction.TIMEOUT.value, "timeout")
        self.assertEqual(TaskAction.MISSING_FILE.value, "missing_file")
        self.assertEqual(TaskAction.INVALID_DATA.value, "invalid_data")
        self.assertEqual(TaskAction.UNRECOVERABLE.value, "unrecoverable")

        self.assertEqual(CriterionType.OUTPUT_EQUALS.value, "output_equals")
        self.assertEqual(CriterionType.OUTPUT_CONTAINS.value, "output_contains")
        self.assertEqual(CriterionType.NON_EMPTY_OUTPUT.value, "non_empty_output")
        self.assertEqual(CriterionType.FILE_EXISTS.value, "file_exists")
        self.assertEqual(CriterionType.DATA_VALID.value, "data_valid")
        self.assertEqual(CriterionType.SERVICE_READY.value, "service_ready")

        self.assertEqual(FailureCategory.NONE.value, "none")
        self.assertEqual(FailureCategory.TRANSIENT_SERVICE_ERROR.value, "transient_service_error")
        self.assertEqual(FailureCategory.TIMEOUT_ERROR.value, "timeout_error")
        self.assertEqual(FailureCategory.MISSING_FILE_ERROR.value, "missing_file_error")
        self.assertEqual(FailureCategory.INVALID_DATA_ERROR.value, "invalid_data_error")
        self.assertEqual(FailureCategory.VERIFICATION_FAILURE.value, "verification_failure")
        self.assertEqual(FailureCategory.UNRECOVERABLE_ERROR.value, "unrecoverable_error")

        self.assertEqual(RecoveryStrategy.RETRY.value, "retry")
        self.assertEqual(RecoveryStrategy.RESET_SERVICE.value, "reset_service")
        self.assertEqual(RecoveryStrategy.RESTORE_FILE.value, "restore_file")
        self.assertEqual(RecoveryStrategy.SANITIZE_DATA.value, "sanitize_data")
        self.assertEqual(RecoveryStrategy.ESCALATE.value, "escalate")

        self.assertEqual(TaskStatus.SUCCEEDED.value, "succeeded")
        self.assertEqual(TaskStatus.RECOVERED.value, "recovered")
        self.assertEqual(TaskStatus.ESCALATED.value, "escalated")
        self.assertEqual(TaskStatus.FAILED.value, "failed")

        self.assertEqual(WorkflowStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkflowStatus.COMPLETED_WITH_RECOVERY.value, "completed_with_recovery")
        self.assertEqual(WorkflowStatus.ESCALATED.value, "escalated")
        self.assertEqual(WorkflowStatus.FAILED.value, "failed")

    def test_success_criteria_validation(self) -> None:
        """Test strict validation of SuccessCriteria definitions."""
        # Non-CriterionType
        with self.assertRaises(TypeError):
            SuccessCriteria("invalid")  # type: ignore

        # FILE_EXISTS validation
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.FILE_EXISTS, "")
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.FILE_EXISTS, "   ")
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.FILE_EXISTS, None)

        valid_fc = SuccessCriteria(CriterionType.FILE_EXISTS, "valid/path.txt")
        self.assertEqual(valid_fc.expected, "valid/path.txt")

        # OUTPUT_CONTAINS validation
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.OUTPUT_CONTAINS, None)

        # DATA_VALID validation
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.DATA_VALID, None)
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.DATA_VALID, [])
        with self.assertRaises(ValueError):
            SuccessCriteria(CriterionType.DATA_VALID, {})

        valid_dv = SuccessCriteria(CriterionType.DATA_VALID, ["id", "status"])
        self.assertEqual(valid_dv.expected, ["id", "status"])

    def test_task_validation(self) -> None:
        """Scenario 13: Task validation (empty id/name, bounds, types)."""
        # Valid task
        t = Task(id="t1", name="Task 1", action=TaskAction.SUCCESS)
        self.assertEqual(t.id, "t1")
        self.assertEqual(t.max_attempts, 3)
        self.assertEqual(t.max_recoveries, 2)
        self.assertTrue(t.idempotent)
        self.assertEqual(t.success_criteria.criterion_type, CriterionType.NON_EMPTY_OUTPUT)

        # Empty id or whitespace
        with self.assertRaises(ValueError):
            Task(id="", name="T", action=TaskAction.SUCCESS)
        with self.assertRaises(ValueError):
            Task(id="   ", name="T", action=TaskAction.SUCCESS)

        # Empty name
        with self.assertRaises(ValueError):
            Task(id="t1", name="", action=TaskAction.SUCCESS)
        with self.assertRaises(ValueError):
            Task(id="t1", name="   ", action=TaskAction.SUCCESS)

        # Invalid action type
        with self.assertRaises(TypeError):
            Task(id="t1", name="T", action="success")  # type: ignore

        # Invalid parameters type
        with self.assertRaises(TypeError):
            Task(id="t1", name="T", action=TaskAction.SUCCESS, parameters=["not", "dict"])  # type: ignore

        # Invalid success_criteria type
        with self.assertRaises(TypeError):
            Task(id="t1", name="T", action=TaskAction.SUCCESS, success_criteria="valid")  # type: ignore

        # Bounds: max_attempts < 1
        with self.assertRaises(ValueError):
            Task(id="t1", name="T", action=TaskAction.SUCCESS, max_attempts=0)
        with self.assertRaises(ValueError):
            Task(id="t1", name="T", action=TaskAction.SUCCESS, max_attempts=-1)

        # Bounds: max_recoveries < 0
        with self.assertRaises(ValueError):
            Task(id="t1", name="T", action=TaskAction.SUCCESS, max_recoveries=-1)

        # max_recoveries=0 is valid (no recoveries permitted)
        t_zero = Task(id="t1", name="T", action=TaskAction.SUCCESS, max_recoveries=0)
        self.assertEqual(t_zero.max_recoveries, 0)

    def test_workflow_validation(self) -> None:
        """Scenario 12: Workflow validation (empty tasks, duplicate IDs, negative budget, empty name)."""
        t1 = Task(id="t1", name="T1", action=TaskAction.SUCCESS)
        t2 = Task(id="t2", name="T2", action=TaskAction.SUCCESS)

        # Valid workflow
        wf = Workflow(name="wf1", tasks=(t1, t2), recovery_budget=4)
        self.assertEqual(wf.name, "wf1")
        self.assertEqual(len(wf.tasks), 2)
        self.assertEqual(wf.recovery_budget, 4)

        # Accepts list and converts to tuple
        wf_list = Workflow(name="wf1", tasks=[t1, t2])
        self.assertIsInstance(wf_list.tasks, tuple)

        # Empty name
        with self.assertRaises(ValueError):
            Workflow(name="", tasks=(t1,))
        with self.assertRaises(ValueError):
            Workflow(name="   ", tasks=(t1,))

        # Empty tasks with EXACT required error message
        with self.assertRaises(ValueError) as ctx:
            Workflow(name="wf", tasks=())
        self.assertEqual(str(ctx.exception), "Workflow must contain at least one task")

        with self.assertRaises(ValueError) as ctx:
            Workflow(name="wf", tasks=[])
        self.assertEqual(str(ctx.exception), "Workflow must contain at least one task")

        # Duplicate task IDs
        t1_dup = Task(id="t1", name="Duplicate T1", action=TaskAction.SUCCESS)
        with self.assertRaises(ValueError) as ctx:
            Workflow(name="wf", tasks=(t1, t1_dup))
        self.assertIn("Duplicate task id 't1'", str(ctx.exception))

        # Negative recovery budget
        with self.assertRaises(ValueError):
            Workflow(name="wf", tasks=(t1,), recovery_budget=-1)

        # Non-Task item
        with self.assertRaises(TypeError):
            Workflow(name="wf", tasks=(t1, "not_a_task"))  # type: ignore

    def test_dataclass_timestamps_and_outcomes(self) -> None:
        """Verify dataclass default factories and TaskOutcome property."""
        ex = ExecutionResult(
            task_id="t1",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value={"ok": True},
            exception_raised=False,
            error_code=FailureCategory.NONE,
            error_message=None,
        )
        self.assertIsInstance(ex.timestamp, str)

        vr = VerificationResult(
            task_id="t1",
            attempt_number=1,
            criterion_type=CriterionType.NON_EMPTY_OUTPUT,
            passed=True,
            reason="passed",
            observed_value={"ok": True},
        )
        self.assertIsInstance(vr.timestamp, str)

        ra = RecoveryAttempt(
            task_id="t1",
            recovery_number=1,
            failure_category=FailureCategory.TIMEOUT_ERROR,
            strategy=RecoveryStrategy.RETRY,
            applied=True,
            details="Retrying",
            escalated=False,
        )
        self.assertIsInstance(ra.timestamp, str)

        outcome_succeeded = TaskOutcome(
            task_id="t1",
            task_name="T1",
            status=TaskStatus.SUCCEEDED,
            attempts_used=1,
            recoveries_used=0,
            final_output={"ok": True},
            escalated=False,
            escalation_reason=None,
        )
        self.assertTrue(outcome_succeeded.succeeded)

        outcome_recovered = TaskOutcome(
            task_id="t1",
            task_name="T1",
            status=TaskStatus.RECOVERED,
            attempts_used=2,
            recoveries_used=1,
            final_output={"ok": True},
            escalated=False,
            escalation_reason=None,
        )
        self.assertTrue(outcome_recovered.succeeded)

        outcome_failed = TaskOutcome(
            task_id="t1",
            task_name="T1",
            status=TaskStatus.FAILED,
            attempts_used=3,
            recoveries_used=2,
            final_output=None,
            escalated=False,
            escalation_reason=None,
        )
        self.assertFalse(outcome_failed.succeeded)

        outcome_escalated = TaskOutcome(
            task_id="t1",
            task_name="T1",
            status=TaskStatus.ESCALATED,
            attempts_used=1,
            recoveries_used=0,
            final_output=None,
            escalated=True,
            escalation_reason="Unrecoverable",
        )
        self.assertFalse(outcome_escalated.succeeded)


if __name__ == "__main__":
    unittest.main()

