"""Command-line demonstration for RecoverOps AI Phase 1."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .failure_simulator import FailureSimulator
from .models import (
    CriterionType,
    SuccessCriteria,
    Task,
    TaskAction,
    Workflow,
)
from .storage import Storage
from .verification import Verifier
from .workflow import WorkflowExecutor


def main() -> None:
    """Run successful, recovered, and safely escalated demonstrations."""

    with tempfile.TemporaryDirectory(prefix="recoverops-demo-") as temp_dir:
        root = Path(temp_dir)
        database_path = root / "demo.sqlite3"
        with Storage(database_path) as storage:
            simulator = FailureSimulator(root)
            verifier = Verifier(root)
            executor = WorkflowExecutor(simulator=simulator, verifier=verifier, storage=storage)

            successful = Workflow(
                name="normal-success",
                tasks=(
                    Task(
                        id="prepare",
                        name="Prepare result",
                        action=TaskAction.SUCCESS,
                        parameters={"output": {"ok": True}},
                        success_criteria=SuccessCriteria(
                            CriterionType.OUTPUT_EQUALS, {"ok": True}
                        ),
                    ),
                ),
            )
            recovered = Workflow(
                name="transient-recovery",
                tasks=(
                    Task(
                        id="service-call",
                        name="Call transient service",
                        action=TaskAction.TRANSIENT_SERVICE,
                        parameters={"failures_before_success": 1},
                        success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
                    ),
                ),
            )
            unrecoverable = Workflow(
                name="human-escalation",
                tasks=(
                    Task(
                        id="unsafe-error",
                        name="Demonstrate an unrecoverable error",
                        action=TaskAction.UNRECOVERABLE,
                        success_criteria=SuccessCriteria(CriterionType.OUTPUT_NOT_EMPTY),
                    ),
                ),
            )

            for workflow in (successful, recovered, unrecoverable):
                result = executor.execute(workflow)
                print(f"\n=== {workflow.name} ===")
                for event in result.timeline:
                    print(event)
                print(f"Final workflow status: {result.status.value}")
                print(
                    "Persisted records: "
                    f"executions={storage.count('task_outcomes', workflow.id)}, "
                    f"recoveries={storage.count('recovery_attempts', workflow.id)}, "
                    f"verifications={storage.count('verification_results', workflow.id)}"
                )


if __name__ == "__main__":
    main()
