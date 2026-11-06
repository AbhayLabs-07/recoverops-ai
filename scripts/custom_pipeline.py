"""Realistic simulated pipeline demonstration: config-sync-and-publish.

Honest Scope:
  Phase 1 executes only simulated tasks through FailureSimulator's closed action set,
  so this is a realistic SIMULATED pipeline demonstrating self-healing, declarative
  verification, and safety guardrails. Wrapping genuine production tasks (e.g., live
  cloud APIs, real HTTP requests, live shell executions) requires the Phase 2
  pluggable task-runner extension point, which is NOT implemented in Phase 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Any

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    SuccessCriteria,
    Task,
    TaskAction,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
)
from app.recovery import RecoveryManager
from app.storage import Storage
from app.verification import Verifier
from app.workflow import WorkflowExecutor


def build_pipeline_tasks(scenario: str) -> tuple[Task, ...]:
    """Construct the 5 pipeline tasks based on the selected operational scenario."""
    if scenario == "happy":
        # All tasks execute and verify cleanly on attempt 1
        t1 = Task(
            id="fetch-remote-config",
            name="Fetch Remote Service Configuration",
            action=TaskAction.SUCCESS,
            parameters={"output": {"version": "2.4.0", "status": "synced"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"version": "2.4.0"}),
            max_attempts=3,
            max_recoveries=2,
        )
        t2 = Task(
            id="restore-local-cache",
            name="Restore Local Configuration Cache",
            action=TaskAction.SUCCESS,
            parameters={
                "create_file": "cache/config_cache.json",
                "file_content": '{"cached": true, "version": "2.4.0"}',
            },
            success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, expected="cache/config_cache.json"),
            max_attempts=3,
            max_recoveries=2,
        )
        t3 = Task(
            id="validate-config-schema",
            name="Validate Runtime Configuration Schema",
            action=TaskAction.SUCCESS,
            parameters={"output": {"version": "2.4.0", "environment": "production", "valid": True}},
            success_criteria=SuccessCriteria(
                CriterionType.DATA_VALID, expected=["version", "environment", "valid"]
            ),
            max_attempts=3,
            max_recoveries=2,
        )
        t4 = Task(
            id="warm-up-service",
            name="Warm Up Backing Service Health Probe",
            action=TaskAction.SUCCESS,
            parameters={"service_ready": True, "output": {"service": "online", "healthy": True}},
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
            max_attempts=3,
            max_recoveries=2,
        )
        t5 = Task(
            id="publish-config",
            name="Publish Live Configuration to Fleet",
            action=TaskAction.SUCCESS,
            parameters={"output": {"published": True, "active_version": "2.4.0"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"published": True}),
            idempotent=False,  # Critical non-idempotent operation
            max_attempts=1,
            max_recoveries=0,
        )
        return (t1, t2, t3, t4, t5)

    elif scenario == "healing":
        # Tasks 1-4 encounter simulated recoverable failures and heal sequentially
        t1 = Task(
            id="fetch-remote-config",
            name="Fetch Remote Service Configuration",
            action=TaskAction.TIMEOUT,
            parameters={
                "succeed_on_attempt": 2,
                "error_message": "Gateway timeout fetching remote configuration from cluster edge",
                "output": {"version": "2.4.0", "status": "synced"},
            },
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"version": "2.4.0"}),
            max_attempts=3,
            max_recoveries=2,
        )
        t2 = Task(
            id="restore-local-cache",
            name="Restore Local Configuration Cache",
            action=TaskAction.MISSING_FILE,
            parameters={
                "file_path": "cache/config_cache.json",
                "restore_content": '{"cached": true, "version": "2.4.0"}',
                "error_message": "Local config cache file missing: cache/config_cache.json",
            },
            success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, expected="cache/config_cache.json"),
            max_attempts=3,
            max_recoveries=2,
        )
        t3 = Task(
            id="validate-config-schema",
            name="Validate Runtime Configuration Schema",
            action=TaskAction.INVALID_DATA,
            parameters={
                "data": {"version": "2.4.0", "invalid_flag": True},
                "sanitized_data": {"version": "2.4.0", "environment": "production", "valid": True},
                "error_message": "Configuration schema missing mandatory 'environment' and 'valid' fields",
            },
            success_criteria=SuccessCriteria(
                CriterionType.DATA_VALID, expected=["version", "environment", "valid"]
            ),
            max_attempts=3,
            max_recoveries=2,
        )
        t4 = Task(
            id="warm-up-service",
            name="Warm Up Backing Service Health Probe",
            action=TaskAction.TRANSIENT_SERVICE,
            parameters={
                "succeed_on_attempt": 2,
                "error_message": "Transient HTTP 503 Service Unavailable on service warmup port 8080",
            },
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
            max_attempts=3,
            max_recoveries=2,
        )
        t5 = Task(
            id="publish-config",
            name="Publish Live Configuration to Fleet",
            action=TaskAction.SUCCESS,
            parameters={"output": {"published": True, "active_version": "2.4.0"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"published": True}),
            idempotent=False,
            max_attempts=1,
            max_recoveries=0,
        )
        return (t1, t2, t3, t4, t5)

    elif scenario == "unsafe":
        # Task 5 (non-idempotent) encounters a failure, triggering immediate escalation
        t1 = Task(
            id="fetch-remote-config",
            name="Fetch Remote Service Configuration",
            action=TaskAction.SUCCESS,
            parameters={"output": {"version": "2.4.0", "status": "synced"}},
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"version": "2.4.0"}),
        )
        t2 = Task(
            id="restore-local-cache",
            name="Restore Local Configuration Cache",
            action=TaskAction.SUCCESS,
            parameters={
                "create_file": "cache/config_cache.json",
                "file_content": '{"cached": true, "version": "2.4.0"}',
            },
            success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, expected="cache/config_cache.json"),
        )
        t3 = Task(
            id="validate-config-schema",
            name="Validate Runtime Configuration Schema",
            action=TaskAction.SUCCESS,
            parameters={"output": {"version": "2.4.0", "environment": "production", "valid": True}},
            success_criteria=SuccessCriteria(
                CriterionType.DATA_VALID, expected=["version", "environment", "valid"]
            ),
        )
        t4 = Task(
            id="warm-up-service",
            name="Warm Up Backing Service Health Probe",
            action=TaskAction.SUCCESS,
            parameters={"service_ready": True, "output": {"service": "online", "healthy": True}},
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
        )
        t5 = Task(
            id="publish-config",
            name="Publish Live Configuration to Fleet",
            action=TaskAction.TIMEOUT,  # Simulated timeout on non-idempotent broadcast
            parameters={
                "succeed_on_attempt": 2,
                "error_message": "Fleet publish socket timeout during irreversible config activation",
            },
            success_criteria=SuccessCriteria(CriterionType.OUTPUT_CONTAINS, expected={"published": True}),
            idempotent=False,  # CRITICAL: Non-idempotent task must never be retried blindly
            max_attempts=3,
            max_recoveries=2,
        )
        return (t1, t2, t3, t4, t5)

    else:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: happy, healing, unsafe.")


def run_pipeline(
    scenario: str = "healing",
    recovery_budget: int = 5,
    db_path: str | Path | None = None,
) -> tuple[WorkflowResult, dict[str, Any]]:
    """Execute the simulated config-sync-and-publish pipeline."""
    print("=" * 70)
    print(" RecoverOps AI — Custom Simulated Pipeline Demo")
    print(f" Workflow: config-sync-and-publish | Scenario: {scenario.upper()}")
    print(f" Shared Recovery Budget: {recovery_budget}")
    print("=" * 70)

    tasks = build_pipeline_tasks(scenario)
    workflow = Workflow(name="config-sync-and-publish", tasks=tasks, recovery_budget=recovery_budget)

    simulator = FailureSimulator()
    verifier = Verifier()
    recovery_manager = RecoveryManager()

    # Track cleanup of temporary database
    is_temp_db = False
    temp_db_path: Path | None = None
    if db_path is None:
        temp_dir = tempfile.mkdtemp(prefix="recoverops_custom_")
        temp_db_path = Path(temp_dir) / "audit.db"
        actual_db: str | Path = temp_db_path
        is_temp_db = True
    else:
        actual_db = db_path

    storage = Storage(actual_db)
    executor = WorkflowExecutor(simulator, verifier, recovery_manager, storage)

    try:
        result = executor.execute(workflow)

        # Print structured scenario progress
        print(f"\nExecution Summary:")
        print(f"  * Status:              {result.status.value}")
        print(f"  * Succeeded:           {result.succeeded}")
        print(f"  * Escalated:           {result.escalated}")
        print(f"  * Total Recoveries:    {result.total_recoveries_used} / {result.recovery_budget}")

        if scenario == "healing":
            print(f"\nSelf-Healing Recovery Audit (Budget Tracking):")
            budget_remaining = recovery_budget
            for outcome in result.task_outcomes:
                if outcome.recoveries_used > 0:
                    budget_remaining -= outcome.recoveries_used
                    print(
                        f"  * Task '{outcome.task_id}': healed with {outcome.recoveries_used} recovery. "
                        f"Remaining budget: {budget_remaining}"
                    )

        if result.escalated:
            print(f"\n  * Escalation Reason:   {result.escalation_reason}")
            print("\n>>> HUMAN ESCALATION REQUIRED <<<")

        # Query SQLite audit summary table
        conn = storage._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM workflows WHERE workflow_name = ? ORDER BY id DESC LIMIT 1;", (workflow.name,))
        wf_row = dict(cursor.fetchone() or {})

        cursor.execute("SELECT COUNT(*) as count FROM task_executions WHERE workflow_id = ?;", (wf_row.get("id"),))
        exec_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM verifications WHERE workflow_id = ?;", (wf_row.get("id"),))
        verif_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM recovery_attempts WHERE workflow_id = ?;", (wf_row.get("id"),))
        rec_count = cursor.fetchone()["count"]

        audit_summary = {
            "workflow_name": wf_row.get("workflow_name"),
            "workflow_id": wf_row.get("id"),
            "status": wf_row.get("status"),
            "recoveries_used": wf_row.get("total_recoveries_used"),
            "recovery_budget": wf_row.get("recovery_budget"),
            "execution_records": exec_count,
            "verification_records": verif_count,
            "recovery_records": rec_count,
        }

        print("\n" + "=" * 70)
        print(" SQLite Audit Summary Table:")
        print(f"  * Workflow Record:     ID={audit_summary['workflow_id']}, Name='{audit_summary['workflow_name']}'")
        print(f"  * Final State:         Status={audit_summary['status']}, Recoveries={audit_summary['recoveries_used']}/{audit_summary['recovery_budget']}")
        print(f"  * Audit Trace Counts:  Executions={audit_summary['execution_records']}, Verifications={audit_summary['verification_records']}, Recoveries={audit_summary['recovery_records']}")
        print("=" * 70)

        return result, audit_summary

    finally:
        storage.close()
        simulator.cleanup()
        if is_temp_db and temp_db_path is not None:
            try:
                temp_db_path.unlink(missing_ok=True)
                if temp_db_path.parent.exists():
                    temp_db_path.parent.rmdir()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RecoverOps AI Custom Simulated Pipeline (config-sync-and-publish)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="healing",
        choices=["happy", "healing", "unsafe"],
        help="Pipeline scenario to execute: happy (all succeed), healing (recoverable faults heal), unsafe (non-idempotent fault escalates)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="Shared recovery budget for the workflow (default: 5; try --budget 2 with --scenario healing to test budget exhaustion)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional SQLite database file path (default: managed ephemeral database with deterministic cleanup)",
    )
    args = parser.parse_args()

    run_pipeline(scenario=args.scenario, recovery_budget=args.budget, db_path=args.db)


if __name__ == "__main__":
    main()
