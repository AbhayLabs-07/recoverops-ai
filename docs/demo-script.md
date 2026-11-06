# RecoverOps AI: 2.5-Minute Demonstration Script

**Target Duration**: 2 minutes 30 seconds  
**Format**: Multi-track presentation script with `[TERMINAL]`, `[CODE]`, and `[VOICEOVER]` cues.

---

### Segment 1: The Problem & Introduction (0:00 - 0:35)

`[TERMINAL]`
```bash
# Terminal displays clean prompt inside recoverops-ai repository
cd recoverops-ai
```

`[VOICEOVER]`
"When production automation workflows fail, typical orchestrators do one of two things: they either crash immediately—requiring 2 AM manual intervention—or they enter naive retry loops, causing duplicate transactions, retry storms, and silent data corruption. Welcome to RecoverOps AI: a deterministic, offline-first, self-healing workflow engine that safely heals transient faults, independently verifies business state, and halts on unsafe conditions."

---

### Segment 2: Architecture & Clean Foundations (0:35 - 1:00)

`[CODE]`
```python
# app/models.py
task = Task(
    id="health-check",
    name="Health check",
    action=TaskAction.SUCCESS,
    parameters={"output": {"healthy": True}},
    success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"healthy": True}),
)
workflow = Workflow(name="example", tasks=(task,), recovery_budget=3)
```

`[VOICEOVER]`
"Phase 1 of RecoverOps AI runs on pure Python 3.11 standard library with zero third-party dependencies. It enforces strict recovery budgets, non-idempotent task locks, sandboxed filesystem isolation, and independent declarative verification: an execution without an exception is never assumed to be a business success."

---

### Segment 3: Live Execution Demonstration (1:00 - 1:50)

`[TERMINAL]`
```bash
python -m app.main
```

`[VOICEOVER]`
"Let's execute the three core demonstration pipelines in real time."

`[TERMINAL]`
```text
======================================================================
DEMO 1: Normal Ingestion Pipeline (Clean Execution)
--- Workflow Result: normal-ingest-pipeline ---
Status: completed | Succeeded: True | Recoveries Used: 0 / 3
======================================================================
```

`[VOICEOVER]`
"Demo 1 runs a multi-task ingest pipeline. All tasks complete cleanly on their first attempt, verified independently."

`[TERMINAL]`
```text
======================================================================
DEMO 2: Self-Healing Pipeline (Service Reset + File Restoration)
Task 'auth-service-check' attempt 1 executed: error=transient_service_error
Task 'auth-service-check' recovery #1: strategy=reset_service, applied=True
Task 'auth-service-check' attempt 2 verified: passed=True
Task 'load-config-file' recovery #1: strategy=restore_file, applied=True
Status: completed_with_recovery | Recoveries Used: 2 / 4
======================================================================
```

`[VOICEOVER]`
"In Demo 2, an auth microservice goes down and a configuration file is missing. RecoverOps AI classifies the service failure, applies a service reset, restores the missing file inside the sandbox, and independently confirms ground-truth health—consuming two units of its four-unit recovery budget."

`[TERMINAL]`
```text
======================================================================
DEMO 3: Unrecoverable Escalation Pipeline (Safety Guardrail Halt)
Task 'critical-hardware-probe' recovery #1: strategy=escalate, escalated=True
Workflow halted: task 'critical-hardware-probe' escalated.
>>> HUMAN ESCALATION REQUIRED <<<
======================================================================
```

`[VOICEOVER]`
"In Demo 3, a fatal hardware fault is detected. Notice the engine does not waste recovery budget on futile retries. It halts immediately, skips downstream dependent tasks, protects system state, and triggers an explicit human escalation."

---

### Segment 4: Verification Suite & Relational Auditability (1:50 - 2:15)

`[TERMINAL]`
```bash
python -m unittest discover -s tests -v
```

`[VOICEOVER]`
"Our test suite runs 44 comprehensive unit tests verifying all 15 core scenarios, including Windows-safe path traversal rejection, idempotency locks, budget exhaustion bounds, and transaction safety in SQLite."

`[TERMINAL]`
```text
Ran 44 tests in 0.620s
OK
```

---

### Segment 5: Closing Value Proposition (2:15 - 2:30)

`[CODE]`
```python
# Relational SQLite Audit Records
storage.get_workflow_runs()
storage.get_executions(workflow_id)
storage.get_verifications(workflow_id)
storage.get_recoveries(workflow_id)
```

`[VOICEOVER]`
"Every decision, attempt, and verification is persisted in transactional SQLite for complete forensic replay. Deterministic, safe, air-gapped, and mathematically bounded: this is RecoverOps AI."

