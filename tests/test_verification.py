"""Unit tests for independent verification engine."""

from __future__ import annotations

import unittest

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    SuccessCriteria,
    Task,
    TaskAction,
)
from app.verification import Verifier


class TestVerification(unittest.TestCase):
    """Test independent declarative verification criteria and security invariants."""

    def setUp(self) -> None:
        self.sim = FailureSimulator()
        self.verifier = Verifier()

    def tearDown(self) -> None:
        self.sim.cleanup()

    def test_exception_raised_never_counts_as_success(self) -> None:
        """CRITICAL: exception_raised == True must NEVER auto-count as success."""
        task = Task(
            id="t1",
            name="T1",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
        )
        res = ExecutionResult(
            task_id="t1",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value={"healthy": True},
            exception_raised=True,  # Exception occurred!
            error_code=FailureCategory.TRANSIENT_SERVICE_ERROR,
            error_message="Something failed",
        )
        vr = self.verifier.verify(task, res, self.sim)
        self.assertFalse(vr.passed)
        self.assertEqual(vr.reason, "Execution failed with exception prior to verification")

    def test_scenario_6_independent_verification_failure(self) -> None:
        """Scenario 6: SUCCESS action returning {'healthy': False} vs expected {'healthy': True}."""
        task = Task(
            id="t1",
            name="Health check",
            action=TaskAction.SUCCESS,
            parameters={"output": {"healthy": False}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"healthy": True}),
        )
        exec_res = self.sim.execute(task, attempt_number=1)
        self.assertFalse(exec_res.exception_raised)  # No execution exception!
        vr = self.verifier.verify(task, exec_res, self.sim)
        self.assertFalse(vr.passed)  # Verification MUST fail!
        self.assertIn("does not equal expected", vr.reason)

    def test_criterion_output_contains(self) -> None:
        """Verify OUTPUT_CONTAINS across dict, string, and collection combinations."""
        # Dict in dict
        task_dict = Task(
            id="td",
            name="TD",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(
                CriterionType.OUTPUT_CONTAINS, {"status": "ok", "env": "prod"}
            ),
        )
        match_res = ExecutionResult(
            task_id="td",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value={"status": "ok", "env": "prod", "extra": 123},
            exception_raised=False,
            error_code=FailureCategory.NONE,
            error_message=None,
        )
        self.assertTrue(self.verifier.verify(task_dict, match_res, self.sim).passed)

        mismatch_res = ExecutionResult(
            task_id="td",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value={"status": "ok", "env": "dev"},
            exception_raised=False,
            error_code=FailureCategory.NONE,
            error_message=None,
        )
        self.assertFalse(self.verifier.verify(task_dict, mismatch_res, self.sim).passed)

        # String in string
        task_str = Task(
            id="ts",
            name="TS",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, "SUCCESS_TOKEN"),
        )
        str_res = ExecutionResult(
            task_id="ts",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value="Operation finished with SUCCESS_TOKEN and code 0",
            exception_raised=False,
            error_code=FailureCategory.NONE,
            error_message=None,
        )
        self.assertTrue(self.verifier.verify(task_str, str_res, self.sim).passed)

        # List subset
        task_list = Task(
            id="tl",
            name="TL",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, ["apple", "cherry"]),
        )
        list_res = ExecutionResult(
            task_id="tl",
            attempt_number=1,
            action=TaskAction.SUCCESS,
            return_value=["apple", "banana", "cherry"],
            exception_raised=False,
            error_code=FailureCategory.NONE,
            error_message=None,
        )
        self.assertTrue(self.verifier.verify(task_list, list_res, self.sim).passed)

    def test_criterion_non_empty_output(self) -> None:
        """Verify NON_EMPTY_OUTPUT criterion."""
        task = Task(
            id="ne",
            name="NE",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
        )
        # None
        r_none = ExecutionResult("ne", 1, TaskAction.SUCCESS, None, False, FailureCategory.NONE, None)
        self.assertFalse(self.verifier.verify(task, r_none, self.sim).passed)

        # Empty string
        r_empty_str = ExecutionResult("ne", 1, TaskAction.SUCCESS, "", False, FailureCategory.NONE, None)
        self.assertFalse(self.verifier.verify(task, r_empty_str, self.sim).passed)

        # Empty dict
        r_empty_dict = ExecutionResult("ne", 1, TaskAction.SUCCESS, {}, False, FailureCategory.NONE, None)
        self.assertFalse(self.verifier.verify(task, r_empty_dict, self.sim).passed)

        # Non-empty dict
        r_valid = ExecutionResult("ne", 1, TaskAction.SUCCESS, {"k": "v"}, False, FailureCategory.NONE, None)
        self.assertTrue(self.verifier.verify(task, r_valid, self.sim).passed)

    def test_criterion_file_exists(self) -> None:
        """Verify FILE_EXISTS checks existence, regular file, and non-empty content."""
        task = Task(
            id="fe",
            name="FE",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, "data/output.log"),
        )
        dummy_res = ExecutionResult("fe", 1, TaskAction.SUCCESS, {"ok": True}, False, FailureCategory.NONE, None)

        # File does not exist
        self.assertFalse(self.verifier.verify(task, dummy_res, self.sim).passed)

        # File exists but is empty (0 bytes)
        file_path = self.sim.resolve_safe_path("data/output.log")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")
        vr_empty = self.verifier.verify(task, dummy_res, self.sim)
        self.assertFalse(vr_empty.passed)
        self.assertIn("exists but is empty", vr_empty.reason)

        # File exists and is non-empty
        file_path.write_text("logged lines here", encoding="utf-8")
        vr_ok = self.verifier.verify(task, dummy_res, self.sim)
        self.assertTrue(vr_ok.passed)
        self.assertIn("regular file, and contains", vr_ok.reason)

    def test_criterion_data_valid(self) -> None:
        """Verify DATA_VALID schema and required key enforcement."""
        task = Task(
            id="dv",
            name="DV",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.DATA_VALID, ["id", "timestamp", "valid"]),
        )
        # Not a dict
        r_not_dict = ExecutionResult("dv", 1, TaskAction.SUCCESS, "raw string", False, FailureCategory.NONE, None)
        self.assertFalse(self.verifier.verify(task, r_not_dict, self.sim).passed)

        # Missing key
        r_missing = ExecutionResult(
            "dv", 1, TaskAction.SUCCESS, {"id": "123", "valid": True}, False, FailureCategory.NONE, None
        )
        self.assertFalse(self.verifier.verify(task, r_missing, self.sim).passed)

        # Explicitly invalid
        r_invalid_flag = ExecutionResult(
            "dv",
            1,
            TaskAction.SUCCESS,
            {"id": "123", "timestamp": "2026-01-01", "valid": False},
            False,
            FailureCategory.NONE,
            None,
        )
        self.assertFalse(self.verifier.verify(task, r_invalid_flag, self.sim).passed)

        # Valid payload
        r_valid = ExecutionResult(
            "dv",
            1,
            TaskAction.SUCCESS,
            {"id": "123", "timestamp": "2026-01-01", "valid": True},
            False,
            FailureCategory.NONE,
            None,
        )
        self.assertTrue(self.verifier.verify(task, r_valid, self.sim).passed)

    def test_criterion_service_ready(self) -> None:
        """Verify SERVICE_READY independently checks simulator state AND output health."""
        task = Task(
            id="sr",
            name="SR",
            action=TaskAction.SUCCESS,
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
        )

        # Simulator says False, output says True -> MUST FAIL
        r_cheat = ExecutionResult(
            "sr", 1, TaskAction.SUCCESS, {"healthy": True}, False, FailureCategory.NONE, None
        )
        self.assertFalse(self.verifier.verify(task, r_cheat, self.sim).passed)

        # Simulator says True, output says False -> MUST FAIL
        self.sim._service_ready["sr"] = True
        r_bad_output = ExecutionResult(
            "sr", 1, TaskAction.SUCCESS, {"healthy": False}, False, FailureCategory.NONE, None
        )
        self.assertFalse(self.verifier.verify(task, r_bad_output, self.sim).passed)

        # Both agree True -> PASS
        r_both_true = ExecutionResult(
            "sr", 1, TaskAction.SUCCESS, {"healthy": True}, False, FailureCategory.NONE, None
        )
        self.assertTrue(self.verifier.verify(task, r_both_true, self.sim).passed)


if __name__ == "__main__":
    unittest.main()

