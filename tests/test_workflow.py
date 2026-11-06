"""Unit tests for WorkflowExecutor orchestrating all end-to-end failure and recovery scenarios."""

from __future__ import annotations

import unittest

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    SuccessCriteria,
    Task,
    TaskAction,
    TaskStatus,
    Workflow,
    WorkflowStatus,
)
from app.recovery import RecoveryManager
from app.storage import Storage
from app.verification import Verifier
from app.workflow import WorkflowExecutor


class TestWorkflow(unittest.TestCase):
    """Test end-to-end execution, budgets, state transitions, and audit records."""

    def setUp(self) -> None:
        self.sim = FailureSimulator()
        self.storage = Storage(None)
        self.verifier = Verifier()
        self.rm = RecoveryManager()
        self.executor = WorkflowExecutor(
            simulator=self.sim,
            verifier=self.verifier,
            recovery_manager=self.rm,
            storage=self.storage,
        )

    def tearDown(self) -> None:
        self.sim.cleanup()
        self.storage.close()

    def test_scenario_1_single_and_multitask_success(self) -> None:
        """Scenario 1: Single and multi-task success + independent verification + audit persistence."""
        t1 = Task(
            id="t1",
            name="Task 1",
            action=TaskAction.SUCCESS,
            parameters={"output": {"step": 1, "status": "ok"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, {"step": 1}),
        )
        t2 = Task(
            id="t2",
            name="Task 2",
            action=TaskAction.SUCCESS,
            parameters={"output": {"step": 2, "status": "ok"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, {"step": 2}),
        )
        wf = Workflow(name="clean_run", tasks=(t1, t2), recovery_budget=3)
        res = self.executor.execute(wf)

        self.assertTrue(res.succeeded)
        self.assertEqual(res.status, WorkflowStatus.COMPLETED)
        self.assertEqual(res.total_recoveries_used, 0)
        self.assertFalse(res.escalated)
        self.assertEqual(len(res.task_outcomes), 2)
        self.assertEqual(res.task_outcomes[0].status, TaskStatus.SUCCEEDED)
        self.assertEqual(res.task_outcomes[1].status, TaskStatus.SUCCEEDED)

        runs = self.storage.get_workflow_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["succeeded"], 1)

    def test_scenario_2_transient_service_recovery(self) -> None:
        """Scenario 2: Transient service: attempt 1 fails, RESET_SERVICE, attempt 2 succeeds, SERVICE_READY passes."""
        task = Task(
            id="svc",
            name="Service Health",
            action=TaskAction.TRANSIENT_SERVICE,
            parameters={"succeed_on_attempt": 2},
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="transient_service_wf", tasks=(task,), recovery_budget=3)
        res = self.executor.execute(wf)

        self.assertTrue(res.succeeded)
        self.assertEqual(res.status, WorkflowStatus.COMPLETED_WITH_RECOVERY)
        self.assertEqual(res.total_recoveries_used, 1)
        self.assertEqual(res.task_outcomes[0].status, TaskStatus.RECOVERED)
        self.assertEqual(res.task_outcomes[0].attempts_used, 2)
        self.assertEqual(res.task_outcomes[0].recoveries_used, 1)

    def test_scenario_3_timeout_retry(self) -> None:
        """Scenario 3: Timeout: RETRY, attempt 2 succeeds."""
        task = Task(
            id="to",
            name="Network Query",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
            success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="timeout_wf", tasks=(task,), recovery_budget=3)
        res = self.executor.execute(wf)

        self.assertTrue(res.succeeded)
        self.assertEqual(res.status, WorkflowStatus.COMPLETED_WITH_RECOVERY)
        self.assertEqual(res.total_recoveries_used, 1)
        self.assertEqual(res.task_outcomes[0].status, TaskStatus.RECOVERED)
        self.assertEqual(res.task_outcomes[0].attempts_used, 2)

    def test_scenario_4_missing_file_restore(self) -> None:
        """Scenario 4: Missing file: RESTORE_FILE inside sandbox, FILE_EXISTS passes."""
        task = Task(
            id="file_task",
            name="File Loader",
            action=TaskAction.MISSING_FILE,
            parameters={
                "file_path": "data/manifest.json",
                "restore_content": '{"version": 1}',
            },
            success_criteria=SuccessCriteria(
                CriterionType.FILE_EXISTS, "data/manifest.json"
            ),
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="missing_file_wf", tasks=(task,), recovery_budget=3)
        res = self.executor.execute(wf)

        self.assertTrue(res.succeeded)
        self.assertEqual(res.status, WorkflowStatus.COMPLETED_WITH_RECOVERY)
        self.assertEqual(res.total_recoveries_used, 1)
        self.assertEqual(res.task_outcomes[0].status, TaskStatus.RECOVERED)

    def test_scenario_5_invalid_data_sanitize(self) -> None:
        """Scenario 5: Invalid data: SANITIZE_DATA, DATA_VALID passes."""
        task = Task(
            id="clean_data",
            name="Data Sanitization",
            action=TaskAction.INVALID_DATA,
            parameters={
                "data": {"id": "x1", "valid": False},
                "required_keys": ["id", "valid"],
            },
            success_criteria=SuccessCriteria(
                CriterionType.DATA_VALID, ["id", "valid"]
            ),
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="invalid_data_wf", tasks=(task,), recovery_budget=3)
        res = self.executor.execute(wf)

        self.assertTrue(res.succeeded)
        self.assertEqual(res.status, WorkflowStatus.COMPLETED_WITH_RECOVERY)
        self.assertEqual(res.total_recoveries_used, 1)
        self.assertEqual(res.task_outcomes[0].status, TaskStatus.RECOVERED)

    def test_scenario_7_max_attempts_exhausted(self) -> None:
        """Scenario 7: max_attempts=2 on a task needing 3 attempts -> escalation."""
        task = Task(
            id="to_slow",
            name="Slow Timeout Task",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 3},
            max_attempts=2,  # Will fail attempt 1, recover, fail attempt 2, then escalate
            max_recoveries=5,
        )
        wf = Workflow(name="attempts_exhausted_wf", tasks=(task,), recovery_budget=5)
        res = self.executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertTrue(res.escalated)
        self.assertEqual(res.status, WorkflowStatus.ESCALATED)
        self.assertIn("Per-task maximum attempts exhausted", res.escalation_reason or "")

    def test_scenario_8_max_recoveries_exhausted(self) -> None:
        """Scenario 8: max_recoveries=1 repeatedly failing -> escalation."""
        task = Task(
            id="to_never",
            name="Never Succeeds",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 10},
            max_attempts=5,
            max_recoveries=1,  # Only 1 recovery permitted
        )
        wf = Workflow(name="recoveries_exhausted_wf", tasks=(task,), recovery_budget=5)
        res = self.executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertTrue(res.escalated)
        self.assertEqual(res.status, WorkflowStatus.ESCALATED)
        self.assertEqual(res.total_recoveries_used, 1)
        self.assertIn("Per-task maximum recoveries exhausted", res.escalation_reason or "")

    def test_scenario_9_shared_workflow_budget_exhaustion(self) -> None:
        """Scenario 9: recovery_budget=1 multi-task: task 1 consumes it, task 2 cannot recover, workflow escalates."""
        t1 = Task(
            id="t1",
            name="First Failure",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
            max_attempts=3,
            max_recoveries=2,
        )
        t2 = Task(
            id="t2",
            name="Second Failure",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="shared_budget_wf", tasks=(t1, t2), recovery_budget=1)
        res = self.executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertTrue(res.escalated)
        self.assertEqual(res.status, WorkflowStatus.ESCALATED)
        self.assertEqual(res.total_recoveries_used, 1)  # Only 1 used before shared budget exhausted
        self.assertIn("Shared workflow recovery budget exhausted", res.escalation_reason or "")

    def test_scenario_10_unrecoverable_immediate_escalation(self) -> None:
        """Scenario 10: Unrecoverable escalates on attempt 1, no wasted budget, skips downstream tasks."""
        t1 = Task(
            id="crit",
            name="Hardware Failure",
            action=TaskAction.UNRECOVERABLE,
            parameters={"error_message": "Kernel panic"},
        )
        t2 = Task(
            id="downstream",
            name="Should Never Run",
            action=TaskAction.SUCCESS,
        )
        wf = Workflow(name="unrec_wf", tasks=(t1, t2), recovery_budget=5)
        res = self.executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertTrue(res.escalated)
        self.assertEqual(res.status, WorkflowStatus.ESCALATED)
        self.assertEqual(res.total_recoveries_used, 0)  # No budget consumed!
        self.assertEqual(len(res.task_outcomes), 1)  # t2 was never executed
        self.assertIn("Unrecoverable failure category", res.escalation_reason or "")

    def test_scenario_11_non_idempotent_task_escalates(self) -> None:
        """Scenario 11: idempotent=False with recoverable failure -> immediate escalation."""
        task = Task(
            id="charge_card",
            name="Process Payment",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
            idempotent=False,  # Unsafe to retry!
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="non_idempotent_wf", tasks=(task,), recovery_budget=5)
        res = self.executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertTrue(res.escalated)
        self.assertEqual(res.status, WorkflowStatus.ESCALATED)
        self.assertEqual(res.total_recoveries_used, 0)
        self.assertIn("Task is marked non-idempotent", res.escalation_reason or "")

    def test_scenario_15_determinism_across_runs(self) -> None:
        """Scenario 15: Determinism across repeated runs (attempts, strategies, statuses)."""
        task = Task(
            id="to_det",
            name="Deterministic Timeout",
            action=TaskAction.TIMEOUT,
            parameters={"succeed_on_attempt": 2},
            max_attempts=3,
            max_recoveries=2,
        )
        wf = Workflow(name="deterministic_wf", tasks=(task,), recovery_budget=3)

        results = []
        for _ in range(3):
            sim = FailureSimulator()
            exec_engine = WorkflowExecutor(simulator=sim)
            res = exec_engine.execute(wf)
            sim.cleanup()
            results.append(res)

        r0 = results[0]
        for r in results[1:]:
            self.assertEqual(r.status, r0.status)
            self.assertEqual(r.succeeded, r0.succeeded)
            self.assertEqual(r.escalated, r0.escalated)
            self.assertEqual(r.total_recoveries_used, r0.total_recoveries_used)
            self.assertEqual(
                r.task_outcomes[0].attempts_used, r0.task_outcomes[0].attempts_used
            )
            self.assertEqual(
                r.task_outcomes[0].recoveries_used, r0.task_outcomes[0].recoveries_used
            )

    def test_a7_public_api_compatibility(self) -> None:
        """A7: Public API compatibility snippet runs unmodified."""
        task = Task(
            id="health-check",
            name="Health check",
            action=TaskAction.SUCCESS,
            parameters={"output": {"healthy": True}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"healthy": True}),
        )
        workflow = Workflow(name="example", tasks=(task,))
        result = WorkflowExecutor(
            simulator=FailureSimulator(), verifier=Verifier()
        ).execute(workflow)
        self.assertTrue(result.succeeded)


if __name__ == "__main__":
    unittest.main()

