"""Unit tests for Stage 2 safety invariants, new strategies, history dependencies, and baselines."""

import json
from pathlib import Path
import tempfile
import unittest

from app.failure_simulator import FailureSimulator
from app.models import (
    CriterionType,
    ExecutionResult,
    FailureCategory,
    RecoveryAttempt,
    RecoveryStrategy,
    SuccessCriteria,
    Task,
    TaskAction,
    Workflow,
)
from app.recovery import RecoveryManager
from app.verification import Verifier
from app.workflow import WorkflowExecutor
from brain.advised_recovery import AdvisedRecoveryManager
from brain.baselines import (
    DecisionTreeBaseline,
    MajorityClassBaseline,
    StatelessLookupBaseline,
    evaluate_baseline_predictions,
)
from brain.evaluate import parse_model_proposal


class DummyMockAdvisor:
    """Mock advisor for deterministic unit testing of safety invariants."""

    def __init__(self, proposal: str = "ESCALATE", confidence: float = 0.95) -> None:
        self.preset_name = "mock"
        self.proposal = proposal
        self.confidence = confidence
        self.consulted_count = 0

    def propose_strategy(self, observation: dict) -> tuple[str, str, float]:
        self.consulted_count += 1
        return self.proposal, f"Mock rationale for {self.proposal}", self.confidence


class TestStage2SafetyAndBaselines(unittest.TestCase):
    """Verify Stage 2 safety invariants, threshold gating, and baseline behaviors."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage2_safety_")).resolve()
        self.simulator = FailureSimulator(root_dir=self.temp_dir)
        self.verifier = Verifier()

    def tearDown(self) -> None:
        self.simulator.cleanup()

    def test_guardrails_precede_model_on_new_families(self) -> None:
        """Invariant: Non-idempotent task with DEADLOCK or RATE_LIMITED must escalate before model is consulted."""
        task = Task(
            id="t_unsafe_rate",
            name="Non-idempotent API",
            action=TaskAction.RATE_LIMITED,
            idempotent=False,
        )
        mock_advisor = DummyMockAdvisor(proposal="RETRY", confidence=0.99)
        adv_mgr = AdvisedRecoveryManager(advisor=mock_advisor)
        executor = WorkflowExecutor(self.simulator, self.verifier, adv_mgr)

        wf = Workflow("test_unsafe_wf", (task,), recovery_budget=5)
        res = executor.execute(wf)

        self.assertFalse(res.succeeded)
        self.assertEqual(res.status.value, "escalated")
        self.assertEqual(mock_advisor.consulted_count, 0)  # Model NEVER consulted!

    def test_model_cannot_trigger_purge_outside_sandbox(self) -> None:
        """Invariant: PURGE_TEMP purges only sandbox-registered temporary files, never outside sandbox."""
        outside_target = self.temp_dir.parent / "safe_file_outside.txt"
        outside_target.write_text("critical outside data", encoding="utf-8")

        task = Task(id="t_purge", name="Purge Task", action=TaskAction.DISK_QUOTA)
        # Attempting to register path outside sandbox raises ValueError
        with self.assertRaises(ValueError):
            self.simulator.register_sandbox_temp_file("../safe_file_outside.txt")

        # Purge only cleans registered sandbox temp files
        self.simulator.purge_temp(task)
        self.assertTrue(outside_target.exists())
        outside_target.unlink()

    def test_escalate_confidence_threshold_gating(self) -> None:
        """Invariant: When threshold is set, ESCALATE is accepted only if confidence >= threshold."""
        task = Task(
            id="t_svc",
            name="Service Task",
            action=TaskAction.TRANSIENT_SERVICE,
            parameters={"succeed_on_attempt": 2},
            success_criteria=SuccessCriteria(CriterionType.SERVICE_READY),
        )

        # 1. Low confidence ESCALATE (0.40) with threshold 0.80 -> overridden by deterministic RESET_SERVICE
        low_conf_advisor = DummyMockAdvisor(proposal="ESCALATE", confidence=0.40)
        mgr_threshold = AdvisedRecoveryManager(
            advisor=low_conf_advisor, escalate_confidence_threshold=0.80
        )
        attempt = mgr_threshold.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TRANSIENT_SERVICE_ERROR,
            simulator=self.simulator,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=5,
            recovery_number=1,
        )
        self.assertFalse(attempt.escalated)
        self.assertEqual(attempt.strategy, RecoveryStrategy.RESET_SERVICE)

        # 2. High confidence ESCALATE (0.95) with threshold 0.80 -> accepted
        high_conf_advisor = DummyMockAdvisor(proposal="ESCALATE", confidence=0.95)
        mgr_threshold2 = AdvisedRecoveryManager(
            advisor=high_conf_advisor, escalate_confidence_threshold=0.80
        )
        attempt2 = mgr_threshold2.attempt_recovery(
            task=task,
            failure_category=FailureCategory.TRANSIENT_SERVICE_ERROR,
            simulator=self.simulator,
            current_attempt=1,
            task_recoveries_used=0,
            workflow_recoveries_used=0,
            workflow_budget=5,
            recovery_number=1,
        )
        self.assertTrue(attempt2.escalated)
        self.assertEqual(attempt2.strategy, RecoveryStrategy.ESCALATE)

    def test_closed_vocabulary_parsing_stage2(self) -> None:
        """Verify closed vocabulary parser accepts ROLLBACK_REPLAY and PURGE_TEMP, defaulting invalid to ESCALATE."""
        s1, r1 = parse_model_proposal("ROLLBACK_REPLAY | RATIONALE: deadlock resolved")
        self.assertEqual(s1, "ROLLBACK_REPLAY")

        s2, r2 = parse_model_proposal("PURGE_TEMP | RATIONALE: quota cleared")
        self.assertEqual(s2, "PURGE_TEMP")

        s3, r3 = parse_model_proposal("FORMAT_HARD_DRIVE | RATIONALE: unsafe action")
        self.assertEqual(s3, "ESCALATE")

    def test_stateless_lookup_fails_on_history_dependent_cases(self) -> None:
        """Baseline verification: stateless lookup table gets 0% accuracy on history dependencies."""
        episodes = [
            {
                "observation": {
                    "failure_category": "timeout_error",
                    "attempt": 2,
                    "history": [{"attempt": 1, "strategy": "RETRY", "succeeded": False}],
                },
                "strategy": "ESCALATE",
                "is_history_dependent": True,
            },
            {
                "observation": {
                    "failure_category": "transient_service_error",
                    "attempt": 3,
                    "history": [
                        {"attempt": 1, "strategy": "RESET_SERVICE", "succeeded": False},
                        {"attempt": 2, "strategy": "RESET_SERVICE", "succeeded": False},
                    ],
                },
                "strategy": "ESCALATE",
                "is_history_dependent": True,
            },
        ]
        stateless = StatelessLookupBaseline()
        preds = [stateless.predict(e) for e in episodes]
        targets = [e["strategy"] for e in episodes]
        res = evaluate_baseline_predictions(preds, targets, episodes)
        # Stateless lookup always predicts RETRY and RESET_SERVICE respectively, scoring 0%
        self.assertEqual(res["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
