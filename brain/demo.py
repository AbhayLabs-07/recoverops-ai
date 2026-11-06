"""Demonstration entrypoint running Part A pipelines under RecoverBrain advisory oversight."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

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
from brain.advisor import BrainAdvisor
from brain.advised_recovery import AdvisedRecoveryManager


def run_benchmark_200(
    preset_name: str = "base",
    num_workflows: int = 200,
    seed: int = 1000,
    escalate_threshold: float = 0.0,
) -> dict[str, Any]:
    """Benchmark recovery success rates across 200 seeded workflows: Advisor vs Deterministic Baseline."""
    print("\n" + "=" * 70)
    print(f" Workflow-Level Benchmark ({num_workflows} Seeded Workflows)")
    print(f" Configuration: Advisor ({preset_name.upper()}) vs Deterministic-Only Baseline")
    print(f" Escalate Confidence Threshold: {escalate_threshold:.2f}")
    print("=" * 70)

    try:
        advisor = BrainAdvisor(preset_name=preset_name)
    except FileNotFoundError:
        print(f"[Notice] Checkpoint for '{preset_name}' not found. Skipping benchmark.")
        return {}

    baseline_succeeded = 0
    advisor_succeeded = 0
    agreements = 0

    rng = random.Random(seed)

    for i in range(num_workflows):
        wf_seed = rng.randint(10000, 999999)
        wf_rng = random.Random(wf_seed)

        # Generate a synthetic multi-task workflow with varying recoverable and unrecoverable conditions
        scenario = wf_rng.choice([
            "clean",
            "transient_svc",
            "timeout",
            "missing_file",
            "invalid_data",
            "rate_limited",
            "deadlock",
            "unrec",
            "ambiguous_verif",
        ])
        budget = wf_rng.randint(2, 4)

        if scenario == "clean":
            t = Task("t1", "Process Task", TaskAction.SUCCESS, parameters={"output": {"status": "ok"}})
        elif scenario == "transient_svc":
            t = Task(
                "t1",
                "Service Call",
                TaskAction.TRANSIENT_SERVICE,
                parameters={"succeed_on_attempt": 2},
                success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
            )
        elif scenario == "timeout":
            t = Task(
                "t1",
                "Network Fetch",
                TaskAction.TIMEOUT,
                parameters={"succeed_on_attempt": 2},
                success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
            )
        elif scenario == "rate_limited":
            t = Task(
                "t1",
                "Rate Limited API",
                TaskAction.RATE_LIMITED,
                parameters={"succeed_on_attempt": 2},
                success_criteria=SuccessCriteria(CriterionType.NON_EMPTY_OUTPUT),
            )
        elif scenario == "deadlock":
            t = Task(
                "t1",
                "Transactional Commit",
                TaskAction.DEADLOCK,
                parameters={"output": {"transaction": "committed", "replayed": True}},
                success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, expected={"transaction": "committed", "replayed": True}),
            )
        elif scenario == "missing_file":
            t = Task(
                "t1",
                "Read Config",
                TaskAction.MISSING_FILE,
                parameters={"file_path": f"bench_{i}.json"},
                success_criteria=SuccessCriteria(CriterionType.FILE_EXISTS, expected=f"bench_{i}.json"),
            )
        elif scenario == "invalid_data":
            t = Task(
                "t1",
                "Validate Ingest",
                TaskAction.INVALID_DATA,
                parameters={"data": {"raw": 1}},
                success_criteria=SuccessCriteria(CriterionType.DATA_VALID, expected=["id", "valid"]),
            )
        elif scenario == "ambiguous_verif":
            t = Task(
                "t1",
                "Strict Verification",
                TaskAction.SUCCESS,
                parameters={"output": {"code": 500}},
                success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, expected={"code": 200}),
            )
        else:  # unrec
            t = Task("t1", "Hardware Probe", TaskAction.UNRECOVERABLE)

        wf_base = Workflow(f"wf_{i}", (t,), recovery_budget=budget)
        wf_adv = Workflow(f"wf_{i}", (t,), recovery_budget=budget)

        # 1. Run with Deterministic-Only Baseline
        storage_base = Storage(None)
        sim_base = FailureSimulator()
        exec_base = WorkflowExecutor(sim_base, Verifier(), RecoveryManager(), storage_base)
        res_base = exec_base.execute(wf_base)
        sim_base.cleanup()
        storage_base.close()

        # 2. Run with Advised Recovery Manager
        storage_adv = Storage(None)
        sim_adv = FailureSimulator()
        adv_mgr = AdvisedRecoveryManager(
            advisor=advisor,
            storage=storage_adv,
            escalate_confidence_threshold=escalate_threshold,
        )
        exec_adv = WorkflowExecutor(sim_adv, Verifier(), adv_mgr, storage_adv)
        res_adv = exec_adv.execute(wf_adv)
        sim_adv.cleanup()
        storage_adv.close()

        if res_base.succeeded:
            baseline_succeeded += 1
        if res_adv.succeeded:
            advisor_succeeded += 1
        if res_base.status == res_adv.status:
            agreements += 1

    base_rate = (baseline_succeeded / num_workflows) * 100.0
    adv_rate = (advisor_succeeded / num_workflows) * 100.0
    base_esc_rate = ((num_workflows - baseline_succeeded) / num_workflows) * 100.0
    adv_esc_rate = ((num_workflows - advisor_succeeded) / num_workflows) * 100.0
    agreement_rate = (agreements / num_workflows) * 100.0
    unnecessary_esc_rate = max(0.0, base_rate - adv_rate)

    print(f"  * Total Seeded Workflows Evaluated:  {num_workflows}")
    print(f"  * Baseline Recovery Success Rate:    {base_rate:.2f}% ({baseline_succeeded}/{num_workflows})")
    print(f"  * Advisor Recovery Success Rate:     {adv_rate:.2f}% ({advisor_succeeded}/{num_workflows})")
    print(f"  * Baseline Escalation Rate:          {base_esc_rate:.2f}%")
    print(f"  * Advisor Escalation Rate:           {adv_esc_rate:.2f}%")
    print(f"  * Unnecessary Escalation Rate:       {unnecessary_esc_rate:.2f}%")
    print(f"  * Status Agreement Rate:             {agreement_rate:.2f}% ({agreements}/{num_workflows})")
    print("=" * 70)

    return {
        "num_workflows": num_workflows,
        "baseline_success_rate": base_rate,
        "advisor_success_rate": adv_rate,
        "baseline_escalation_rate": base_esc_rate,
        "advisor_escalation_rate": adv_esc_rate,
        "unnecessary_escalation_rate": unnecessary_esc_rate,
        "agreement_rate": agreement_rate,
    }


def run_advised_demos(
    preset_name: str = "base",
    escalate_threshold: float = 0.0,
    run_benchmark: bool = True,
) -> None:
    """Execute the three demonstration pipelines using AdvisedRecoveryManager."""
    print("=" * 70)
    print(f" RecoverBrain Advised Self-Healing Demonstration (Preset: {preset_name.upper()})")
    print(f" Escalate Confidence Threshold: {escalate_threshold:.2f}")
    print("=" * 70)

    storage = Storage(None)
    try:
        advisor = BrainAdvisor(preset_name=preset_name)
    except FileNotFoundError:
        print(f"[Notice] Checkpoint for '{preset_name}' not found. Please train first.")
        return

    advised_recovery_mgr = AdvisedRecoveryManager(
        advisor=advisor,
        storage=storage,
        escalate_confidence_threshold=escalate_threshold,
    )

    # Demo 1: Normal Ingest
    print("\n[Pipeline 1: Normal Ingestion]")
    t1 = Task("extract", "Extract Source", TaskAction.SUCCESS, parameters={"output": {"count": 100}})
    wf1 = Workflow("normal-ingest", (t1,), recovery_budget=3)
    sim1 = FailureSimulator()
    exec1 = WorkflowExecutor(sim1, Verifier(), advised_recovery_mgr, storage)
    res1 = exec1.execute(wf1)
    print(f"Outcome: {res1.status.value} (Recoveries used: {res1.total_recoveries_used})")

    # Demo 2: Self-Healing Service & File
    print("\n[Pipeline 2: Self-Healing Transient Service]")
    t2 = Task(
        "svc-check",
        "Microservice probe",
        TaskAction.TRANSIENT_SERVICE,
        parameters={"succeed_on_attempt": 2, "error_message": "Service auth-gateway temporarily unavailable"},
        success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
    )
    wf2 = Workflow("self-healing-service", (t2,), recovery_budget=3)
    sim2 = FailureSimulator()
    exec2 = WorkflowExecutor(sim2, Verifier(), advised_recovery_mgr, storage)
    res2 = exec2.execute(wf2)
    print(f"Outcome: {res2.status.value} (Recoveries used: {res2.total_recoveries_used})")
    if res2.status == WorkflowStatus.COMPLETED_WITH_RECOVERY:
        print(">>> Self-Healing Pipeline Result: COMPLETED_WITH_RECOVERY (Success) <<<")
    else:
        print(f">>> Self-Healing Pipeline Result: {res2.status.value} (Did not recover) <<<")

    # Demo 3: Unrecoverable Hardware Fault
    print("\n[Pipeline 3: Unrecoverable Fault]")
    t3 = Task("bus-error", "Hardware probe", TaskAction.UNRECOVERABLE)
    wf3 = Workflow("unrec-halt", (t3,), recovery_budget=3)
    sim3 = FailureSimulator()
    exec3 = WorkflowExecutor(sim3, Verifier(), advised_recovery_mgr, storage)
    res3 = exec3.execute(wf3)
    print(f"Outcome: {res3.status.value} (Escalated: {res3.escalated})")

    # Inspect brain_proposals table
    conn = storage._get_connection()
    cursor = conn.execute("SELECT * FROM brain_proposals ORDER BY id ASC;")
    proposals = [dict(row) for row in cursor.fetchall()]

    print("\n" + "=" * 70)
    print(f" Audited RecoverBrain Proposals in SQLite ({len(proposals)} recorded):")
    for p in proposals:
        print(
            f"  * Task '{p['task_id']}': Proposed={p['proposed_strategy']:<14} | "
            f"Final={p['final_strategy']:<14} | Accepted={bool(p['accepted'])}"
        )
        if p["override_reason"]:
            print(f"      Override Reason: {p['override_reason']}")
    print("=" * 70)

    sim1.cleanup()
    sim2.cleanup()
    sim3.cleanup()
    storage.close()

    if run_benchmark:
        run_benchmark_200(preset_name=preset_name, escalate_threshold=escalate_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="RecoverBrain advised self-healing demo")
    parser.add_argument(
        "--preset",
        type=str,
        default=os.environ.get("RECOVERBRAIN_PRESET", "base"),
        choices=["tiny", "small", "base"],
        help="Model preset to load",
    )
    parser.add_argument(
        "--escalate-confidence-threshold",
        type=float,
        default=0.0,
        help="Softmax confidence threshold required to honor an ESCALATE proposal (default: 0.0)",
    )
    parser.add_argument(
        "--no-benchmark",
        dest="run_benchmark",
        action="store_false",
        help="Skip the 200-workflow benchmark",
    )
    args = parser.parse_args()
    run_advised_demos(
        preset_name=args.preset,
        escalate_threshold=args.escalate_confidence_threshold,
        run_benchmark=args.run_benchmark,
    )


if __name__ == "__main__":
    main()
