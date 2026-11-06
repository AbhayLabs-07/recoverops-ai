"""Command-line entrypoint executing end-to-end self-healing demonstrations."""

from __future__ import annotations

import logging
import os
import sys

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    SuccessCriteria,
    Task,
    TaskAction,
    Workflow,
    WorkflowStatus,
)
from app.recovery import RecoveryManager
from app.storage import Storage
from app.verification import Verifier
from app.workflow import WorkflowExecutor

BANNER = r"""
======================================================================
 ____                                ___              _     ___ 
|  _ \ ___  ___ _____   _____ _ __ / _ \ _ __  ___   / \   |_ _|
| |_) / _ \/ __/ _ \ \ / / _ \ '__| | | | '_ \/ __| / _ \   | | 
|  _ <  __/ (_| (_) \ V /  __/ |  | |_| | |_) \__ \/ ___ \  | | 
|_| \_\___|\___\___/ \_/ \___|_|   \___/| .__/|___/_/   \_\|___|
                                        |_|                     
     Deterministic Self-Healing Engine (Phase 1 stdlib-only)
======================================================================
"""


def configure_logging() -> None:
    """Configure standard logging level based on RECOVEROPS_LOG_LEVEL environment variable."""
    raw_level = os.environ.get("RECOVEROPS_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, raw_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_configured_storage() -> Storage:
    """Instantiate Storage according to RECOVEROPS_DB_PATH specification (C1)."""
    raw_path = os.environ.get("RECOVEROPS_DB_PATH", "").strip()
    if not raw_path or raw_path == ":memory:":
        return Storage(None)
    return Storage(raw_path)


def print_workflow_report(result) -> None:
    """Render a structured, formatted summary of the workflow execution."""
    print(f"\n--- Workflow Result: {result.workflow_name} ---")
    print(f"Status:               {result.status.value}")
    print(f"Succeeded:            {result.succeeded}")
    print(f"Escalated:            {result.escalated}")
    if result.escalation_reason:
        print(f"Escalation Reason:    {result.escalation_reason}")
    print(f"Recoveries Used:      {result.total_recoveries_used} / {result.recovery_budget}")
    print("\nTask Outcomes:")
    for outcome in result.task_outcomes:
        print(f"  * Task [{outcome.task_id}] '{outcome.task_name}'")
        print(f"      Status:     {outcome.status.value}")
        print(f"      Attempts:   {outcome.attempts_used}")
        print(f"      Recoveries: {outcome.recoveries_used}")
        if outcome.escalation_reason:
            print(f"      Escalation: {outcome.escalation_reason}")
        if outcome.verifications:
            last_v = outcome.verifications[-1]
            print(f"      Verification: passed={last_v.passed} ({last_v.reason})")

    print("\nExecution Timeline:")
    for entry in result.timeline:
        print(f"  {entry}")

    if result.status == WorkflowStatus.ESCALATED:
        print("\n>>> HUMAN ESCALATION REQUIRED <<<\n")


def run_pipeline_1(storage: Storage) -> None:
    """Demo 1: Normal Ingestion Pipeline (all tasks succeed on attempt 1)."""
    print("\n" + "=" * 70)
    print("DEMO 1: Normal Ingestion Pipeline (Clean Execution)")
    print("=" * 70)

    task1 = Task(
        id="extract-source",
        name="Extract Source Records",
        action=TaskAction.SUCCESS,
        parameters={"output": {"records_count": 100, "status": "extracted"}},
        success_criteria=SuccessCriteria(
            CriterionType.OUTPUT_CONTAINS, {"records_count": 100}
        ),
    )
    task2 = Task(
        id="validate-schema",
        name="Validate Schema",
        action=TaskAction.SUCCESS,
        parameters={
            "output": {"id": "rec_001", "valid": True, "checksum": "abc123xyz"}
        },
        success_criteria=SuccessCriteria(
            CriterionType.DATA_VALID, ["id", "valid", "checksum"]
        ),
    )
    task3 = Task(
        id="export-artifact",
        name="Export Ingestion Artifact",
        action=TaskAction.SUCCESS,
        parameters={
            "create_file": "output/ingestion_report.json",
            "file_content": '{"status": "exported", "count": 100}',
            "output": {"exported": True},
        },
        success_criteria=SuccessCriteria(
            CriterionType.FILE_EXISTS, "output/ingestion_report.json"
        ),
    )

    wf = Workflow(
        name="normal-ingest-pipeline",
        tasks=(task1, task2, task3),
        recovery_budget=3,
    )
    sim = FailureSimulator()
    executor = WorkflowExecutor(
        simulator=sim, verifier=Verifier(), recovery_manager=RecoveryManager(), storage=storage
    )
    result = executor.execute(wf)
    print_workflow_report(result)
    sim.cleanup()


def run_pipeline_2(storage: Storage) -> None:
    """Demo 2: Self-Healing Service & Data Pipeline (recovers from service and file faults)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Self-Healing Pipeline (Service Reset + File Restoration)")
    print("=" * 70)

    task1 = Task(
        id="auth-service-check",
        name="Verify Auth Microservice",
        action=TaskAction.TRANSIENT_SERVICE,
        parameters={
            "succeed_on_attempt": 2,
            "output": {"service": "online", "healthy": True},
        },
        success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
        idempotent=True,
        max_attempts=3,
        max_recoveries=2,
    )
    task2 = Task(
        id="load-config-file",
        name="Load Dynamic Config File",
        action=TaskAction.MISSING_FILE,
        parameters={
            "file_path": "configs/runtime.json",
            "restore_content": '{"cluster_mode": "active", "restored": true}',
        },
        success_criteria=SuccessCriteria(
            CriterionType.FILE_EXISTS, "configs/runtime.json"
        ),
        idempotent=True,
        max_attempts=3,
        max_recoveries=2,
    )

    wf = Workflow(
        name="self-healing-service-pipeline",
        tasks=(task1, task2),
        recovery_budget=4,
    )
    sim = FailureSimulator()
    executor = WorkflowExecutor(
        simulator=sim, verifier=Verifier(), recovery_manager=RecoveryManager(), storage=storage
    )
    result = executor.execute(wf)
    print_workflow_report(result)
    sim.cleanup()


def run_pipeline_3(storage: Storage) -> None:
    """Demo 3: Unrecoverable Escalation Pipeline (escalates without wasting budget)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Unrecoverable Escalation Pipeline (Safety Guardrail Halt)")
    print("=" * 70)

    task1 = Task(
        id="critical-hardware-probe",
        name="Inspect Specialized Hardware Interface",
        action=TaskAction.UNRECOVERABLE,
        parameters={"error_message": "Hardware bus parity error detected"},
        success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
        idempotent=True,
    )
    task2 = Task(
        id="downstream-never-reached",
        name="Downstream Dependent Task",
        action=TaskAction.SUCCESS,
        parameters={"output": {"status": "should_not_run"}},
        success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
    )

    wf = Workflow(
        name="unrecoverable-escalation-pipeline",
        tasks=(task1, task2),
        recovery_budget=5,
    )
    sim = FailureSimulator()
    executor = WorkflowExecutor(
        simulator=sim, verifier=Verifier(), recovery_manager=RecoveryManager(), storage=storage
    )
    result = executor.execute(wf)
    print_workflow_report(result)
    sim.cleanup()


def main() -> None:
    """Run all Phase 1 demonstration scenarios."""
    print(BANNER)
    configure_logging()
    storage = get_configured_storage()
    try:
        run_pipeline_1(storage)
        run_pipeline_2(storage)
        run_pipeline_3(storage)
    finally:
        storage.close()


if __name__ == "__main__":
    main()

