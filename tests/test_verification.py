"""Independent verification tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.models import (
    CriterionType,
    ExecutionResult,
    FailureKind,
    SuccessCriteria,
    Task,
    TaskAction,
    VerificationStatus,
)
from app.verification import Verifier


class VerificationTests(unittest.TestCase):
    def _task(self, criteria: SuccessCriteria) -> Task:
        return Task("task", "Task", TaskAction.SUCCESS, criteria)

    def _execution(self, output: object, succeeded: bool = True) -> ExecutionResult:
        return ExecutionResult(
            workflow_id="workflow",
            task_id="task",
            attempt=1,
            executor_succeeded=succeeded,
            output=output,
            failure_kind=FailureKind.NONE if succeeded else FailureKind.UNKNOWN,
            error=None if succeeded else "action failure",
        )

    def test_output_equality_is_independent(self) -> None:
        result = Verifier().verify(
            self._task(SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"value": 1})),
            self._execution({"value": 1}),
        )
        self.assertEqual(result.status, VerificationStatus.PASSED)

    def test_output_mismatch_fails(self) -> None:
        result = Verifier().verify(
            self._task(SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"value": 1})),
            self._execution({"value": 2}),
        )
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("equal", result.error or "")

    def test_data_validity_rejects_invalid_payload(self) -> None:
        result = Verifier().verify(
            self._task(SuccessCriteria(CriterionType.DATA_VALID)),
            self._execution({"valid": False}),
        )
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertFalse(result.succeeded)

    def test_file_exists_checks_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.txt").write_text("ok", encoding="utf-8")
            task = self._task(SuccessCriteria(CriterionType.FILE_EXISTS, "present.txt"))

            passed = Verifier(root).verify(task, self._execution({"ok": True}))
            missing = Verifier(root).verify(
                self._task(SuccessCriteria(CriterionType.FILE_EXISTS, "missing.txt")),
                self._execution({"ok": True}),
            )

        self.assertEqual(passed.status, VerificationStatus.PASSED)
        self.assertEqual(missing.status, VerificationStatus.FAILED)

    def test_action_failure_can_never_verify_as_success(self) -> None:
        result = Verifier().verify(
            self._task(SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY)),
            self._execution({"ok": True}, succeeded=False),
        )
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertFalse(result.succeeded)

    def test_all_common_empty_containers_fail_non_empty_check(self) -> None:
        task = self._task(SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY))
        for output in ((), set(), b"", bytearray()):
            with self.subTest(output=type(output).__name__):
                result = Verifier().verify(task, self._execution(output))
                self.assertEqual(result.status, VerificationStatus.FAILED)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = self._task(SuccessCriteria(CriterionType.FILE_EXISTS, "../outside"))
            result = Verifier(directory).verify(task, self._execution({"ok": True}))
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("outside", result.error or "")


if __name__ == "__main__":
    unittest.main()
