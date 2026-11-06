"""Unit tests for FailureSimulator sandbox isolation and fault injection."""

from __future__ import annotations

from pathlib import Path
import unittest

from app.failure_simulator import FailureSimulator, UnrestorableFileError
from app.models import FailureCategory, Task, TaskAction


class TestFailureSimulator(unittest.TestCase):
    """Test safe sandbox resolution, deterministic execution, and state remediation."""

    def setUp(self) -> None:
        self.sim = FailureSimulator()

    def tearDown(self) -> None:
        self.sim.cleanup()

    def test_sandbox_path_traversal_rejection(self) -> None:
        """Scenario 4 part 2: simulator.resolve_safe_path('../outside.txt') raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("../outside.txt")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("../../secret.key")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("sub/../../escape.txt")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("/absolute/unix/path")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("C:\\windows\\system32\\cmd.exe")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("D:relative_drive_path.txt")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        with self.assertRaises(ValueError) as ctx:
            self.sim.resolve_safe_path("")
        self.assertEqual(str(ctx.exception), "Path traversal outside simulator root is forbidden")

        # Valid relative paths
        safe_p = self.sim.resolve_safe_path("data/input.json")
        self.assertTrue(str(safe_p).startswith(str(self.sim.root_dir)))

    def test_success_action_with_file_creation(self) -> None:
        """Verify SUCCESS action behavior, output propagation, and file creation."""
        task = Task(
            id="s1",
            name="Success Task",
            action=TaskAction.SUCCESS,
            parameters={
                "output": {"result": 42},
                "create_file": "nested/output.txt",
                "file_content": "deterministic-data",
                "service_ready": True,
            },
        )
        res = self.sim.execute(task, attempt_number=1)
        self.assertFalse(res.exception_raised)
        self.assertEqual(res.return_value, {"result": 42})
        self.assertEqual(res.error_code, FailureCategory.NONE)
        self.assertTrue(self.sim.is_service_ready("s1"))

        target_file = self.sim.resolve_safe_path("nested/output.txt")
        self.assertTrue(target_file.is_file())
        self.assertEqual(target_file.read_text(encoding="utf-8"), "deterministic-data")

    def test_transient_service_reset_precedence(self) -> None:
        """Verify TRANSIENT_SERVICE fault injection, reset remediation, and C3 precedence."""
        task = Task(
            id="svc1",
            name="Service Task",
            action=TaskAction.TRANSIENT_SERVICE,
            parameters={"succeed_on_attempt": 3},
        )
        # Attempt 1: not ready
        res1 = self.sim.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.TRANSIENT_SERVICE_ERROR)

        # Mutate via reset_service -> next attempt succeeds immediately
        desc = self.sim.reset_service(task)
        self.assertIn("Service state reset", desc)
        self.assertTrue(self.sim.is_service_ready("svc1"))

        res2 = self.sim.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertEqual(res2.error_code, FailureCategory.NONE)
        self.assertEqual(res2.return_value, {"service": "online", "healthy": True})

    def test_timeout_action(self) -> None:
        """Verify TIMEOUT action fails until succeed_on_attempt is reached."""
        task = Task(
            id="to1",
            name="Timeout Task",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
        )
        res1 = self.sim.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.TIMEOUT_ERROR)
        self.assertEqual(res1.error_message, "Simulated deterministic operation timeout")

        res2 = self.sim.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertEqual(res2.error_code, FailureCategory.NONE)
        self.assertEqual(res2.return_value, {"completed": True, "timed_out": False})

    def test_missing_file_and_restore(self) -> None:
        """Verify MISSING_FILE detection and restore_file mutator."""
        task = Task(
            id="mf1",
            name="Missing File Task",
            action=TaskAction.MISSING_FILE,
            parameters={
                "file_path": "data/required.json",
                "restore_content": {"restored": True, "source": "backup"},
            },
        )
        res1 = self.sim.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.MISSING_FILE_ERROR)
        self.assertEqual(res1.error_message, "Required file missing: data/required.json")

        # Mutate via restore_file
        desc = self.sim.restore_file(task)
        self.assertIn("successfully restored", desc)

        res2 = self.sim.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertEqual(res2.error_code, FailureCategory.NONE)
        self.assertEqual(res2.return_value, {"restored": True, "source": "backup"})

    def test_unrestorable_file_error(self) -> None:
        """Verify UnrestorableFileError when restoration is explicitly disabled."""
        task = Task(
            id="mf2",
            name="Unrestorable File",
            action=TaskAction.MISSING_FILE,
            parameters={"file_path": "lost.txt", "unrestorable": True},
        )
        with self.assertRaises(UnrestorableFileError):
            self.sim.restore_file(task)

    def test_invalid_data_and_sanitize(self) -> None:
        """Verify INVALID_DATA detection and sanitize_data mutator."""
        task = Task(
            id="inv1",
            name="Invalid Data Task",
            action=TaskAction.INVALID_DATA,
            parameters={
                "data": {"id": "item_1", "valid": False, "corrupt_field": None},
                "required_keys": ["id", "valid"],
                "sanitized_data": {"id": "item_1", "valid": True, "clean": True},
            },
        )
        res1 = self.sim.execute(task, attempt_number=1)
        self.assertTrue(res1.exception_raised)
        self.assertEqual(res1.error_code, FailureCategory.INVALID_DATA_ERROR)

        desc = self.sim.sanitize_data(task)
        self.assertIn("Payload sanitized", desc)

        res2 = self.sim.execute(task, attempt_number=2)
        self.assertFalse(res2.exception_raised)
        self.assertEqual(res2.error_code, FailureCategory.NONE)
        self.assertEqual(res2.return_value, {"id": "item_1", "valid": True, "clean": True})

    def test_unrecoverable_action(self) -> None:
        """Verify UNRECOVERABLE action always returns UNRECOVERABLE_ERROR."""
        task = Task(
            id="unrec1",
            name="Unrecoverable Task",
            action=TaskAction.UNRECOVERABLE,
            parameters={"error_message": "Custom hardware fault"},
        )
        res = self.sim.execute(task, attempt_number=1)
        self.assertTrue(res.exception_raised)
        self.assertEqual(res.error_code, FailureCategory.UNRECOVERABLE_ERROR)
        self.assertEqual(res.error_message, "Custom hardware fault")


if __name__ == "__main__":
    unittest.main()

