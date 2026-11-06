# RecoverOps AI

> **Deterministic, Offline-First, Zero-Dependency Self-Healing Workflow Engine**  
> *Phase 1 Core Engine & RecoverBrain Transformer Architecture*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20third--party%20(stdlib%20only)-success.svg)]()
[![Security](https://img.shields.io/badge/security-sandbox%20isolated-blueviolet.svg)]()

---

## 1. Overview & Context

**RecoverOps AI** is an industrial-grade, offline-first orchestration engine built for mission-critical automation pipelines. When complex automations fail in production—due to transient networking hiccups, missing file dependencies, poisoned data payloads, or unrecoverable hardware faults—typical systems either crash abruptly or blindly loop through naive retries, exhausting system resources and causing silent data corruption.

RecoverOps AI eliminates naive failure handling by embedding a **deterministic self-healing control loop**:
1. **Detect & Classify**: Pinpoints exact domain failure categories across a closed fault taxonomy.
2. **Bounded Recovery**: Applies deterministic remediations bounded by strict per-task limits and a shared workflow recovery budget.
3. **Independent Verification**: Evaluates declarative success criteria against actual domain state, refusing to trust mere process exit codes.
4. **Relational Audit Trail**: Records every execution attempt, verification check, and remediation decision in transactional SQLite using strict parameterized SQL.
5. **Safety Escalation**: Immediately halts downstream work and escalates to human operators when encountering non-idempotent tasks, exhausted budgets, or unrecoverable hardware faults.

---

## 2. Why Naive Retries Fail

In enterprise automation, naive retry loops (such as blind exponential backoff) introduce severe operational risks:
- **State Corruption on Non-Idempotent Actions**: Retrying an operation that partially mutated a database or charged a credit card can result in duplicate payments or invalid data schemas.
- **Cascading Outages (Retry Storms)**: When a downstream microservice is degraded, hundreds of workers retrying concurrently amplify traffic, preventing the service from recovering.
- **Resource Exhaustion**: Loops without global budgets burn CPU cycles, disk I/O, and worker threads on hopeless tasks.
- **Blind Faith in Exit Status**: An operation that exits with return code `0` while outputting a blank file or corrupted JSON will be falsely marked as succeeded by naive orchestrators.

RecoverOps AI replaces naive retries with **causal classification**, **idempotency gating**, and **independent declarative verification**.

---

## 3. Phase 1 Architecture & Execution Flow

```
                      +----------------------------------------+
                      |            Workflow Definition         |
                      |   (Ordered Tasks + Recovery Budget)    |
                      +-------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              WorkflowExecutor Loop                                |
|                                                                                   |
|    +-----------------------+     Execution      +----------------------------+    |
|    |    FailureSimulator   | -----------------> |      ExecutionResult       |    |
|    |   (Sandbox Enclosed)  |                    +-------------+--------------+    |
|    +-----------------------+                                  |                   |
|                                                               v                   |
|    +-----------------------+     Evaluate       +----------------------------+    |
|    |        Verifier       | <----------------- |     Independent State      |    |
|    | (Declarative Criteria)|                    |        Verification        |    |
|    +-----------------------+                    +-------------+--------------+    |
|                                                               |                   |
|                                         +---------------------+-----------------+ |
|                                         |                                       | |
|                                   Passed=True                             Passed=False    |
|                                         |                                       | |
|                                         v                                       v |
|                              +--------------------+                 +---------------+
|                              | Mark SUCCEEDED or  |                 | Classify Fault|
|                              |     RECOVERED      |                 |   Category    |
|                              +--------------------+                 +-------+-------+
|                                                                             |     |
|                                         +-----------------------------------+     |
|                                         v                                         |
|    +--------------------------------------------------------------------------+   |
|    |                      RecoveryManager Guardrail Pipeline                  |   |
|    |                                                                          |   |
|    |  [1. Unrecoverable?] ---> Yes ---> ESCALATE (Applied=False, Halt)        |   |
|    |  [2. Non-idempotent?] --> Yes ---> ESCALATE (Applied=False, Halt)        |   |
|    |  [3. Task Attempts?] ---> Exceeded -> ESCALATE (Applied=False, Halt)     |   |
|    |  [4. Task Recoveries?] -> Exceeded -> ESCALATE (Applied=False, Halt)     |   |
|    |  [5. Shared Budget?] ---> Exceeded -> ESCALATE (Applied=False, Halt)     |   |
|    |  [6. Select Strategy] --> RETRY | RESET_SERVICE | RESTORE_FILE | SANITIZE|   |
|    +------------------------------------+-------------------------------------+   |
|                                         |                                         |
|                                Remediation Applied                                |
|                                (Budget Incremented)                               |
|                                         |                                         |
|                                         +-----------------------------------------+
+-----------------------------------------------------------------------------------+
                                          |
                                          v
                      +----------------------------------------+
                      |         Relational Audit Store         |
                      |   (SQLite, Foreign Keys, Parameterized)|
                      +----------------------------------------+
```

---

## 4. Scope Boundaries

| Capability | Phase 1 Core Engine (app/) | RecoverBrain (brain/) |
| :--- | :--- | :--- |
| **Runtime Dependencies** | **Zero** (Pure Python 3.11 Standard Library) | PyTorch, NumPy |
| **Execution Environment**| Air-gapped, Offline-First, Local | RTX 5050 Laptop GPU / CUDA |
| **Process Model** | Single-threaded, Synchronous, Deterministic | PyTorch Causal Transformer |
| **Network / Sockets** | **Forbidden** (Zero socket/HTTP/cloud imports) | Forbidden (No remote APIs) |
| **File Access** | Strict path-jailed sandbox (`resolve_safe_path`) | Strict path-jailed `brain/sandbox/`|
| **SQL Engine** | Parameterized SQLite (`PRAGMA foreign_keys = ON`) | Parameterized SQLite (`brain_proposals`)|
| **Recovery Decision** | **Final Decision Maker** (Owns all budgets) | Advisory proposal only |

---

## 5. Security & Isolation Guarantees

1. **Path-Jailed Sandbox**:
   Every filesystem interaction inside `FailureSimulator` passes through `resolve_safe_path()`. It rejects absolute paths, Windows drive specifiers (`C:`, `D:`), null bytes, and parent directory traversal tokens (`..`). Escapes outside the designated root raise `ValueError: Path traversal outside simulator root is forbidden`.
2. **Parameterized SQL Only**:
   The relational audit store strictly enforces SQLite `?` parameter substitution. No format strings (`%` or `f"{}"`) are ever executed against the database engine.
3. **Prohibition of Dynamic Execution**:
   Phase 1 code contains zero calls to `eval()`, `exec()`, `compile()`, `__import__()`, `pickle`, or `subprocess`/`os.system`.
4. **Idempotency Guardrails**:
   Tasks flagged with `idempotent=False` are never automatically retried or recovered, safeguarding transactional business logic from repeated execution.

---

## 6. Installation & Verification

### Prerequisites
- Python 3.11 installed (`.python-version` is set to `3.11`).
- No virtual environment required for Phase 1 (stdlib only).

### Running Tests
Execute the comprehensive 44-test verification suite:

```bash
# Linux / macOS / Windows PowerShell
python -m unittest discover -s tests -v
```

### Running Demonstration Workflows
Run the end-to-end self-healing pipeline scenarios:

```bash
python -m app.main
```

### Using Make
```bash
make test         # Run unittest suite
make run          # Run demonstration pipelines
make clean        # Remove caches and ephemeral databases
```

### Running with Docker
```bash
docker build -t recoverops-ai:phase1 .
docker run --rm recoverops-ai:phase1
```

---

## 7. Public API Quickstart

```python
from app.failure_simulator import FailureSimulator
from app.models import CriterionType, SuccessCriteria, Task, TaskAction, Workflow
from app.verification import Verifier
from app.workflow import WorkflowExecutor

# 1. Define tasks with explicit declarative criteria
task = Task(
    id="health-check",
    name="Health check",
    action=TaskAction.SUCCESS,
    parameters={"output": {"healthy": True}},
    success_criteria=SuccessCriteria(CriterionType.OUTPUT_EQUALS, {"healthy": True}),
)

# 2. Bundle into a workflow with bounded recovery budget
workflow = Workflow(name="example", tasks=(task,), recovery_budget=3)

# 3. Execute with deterministic orchestration
executor = WorkflowExecutor(simulator=FailureSimulator(), verifier=Verifier())
result = executor.execute(workflow)

assert result.succeeded
print(f"Workflow finished: status={result.status.value}")
```

---

## 8. Environmental Configuration

RecoverOps AI reads optional settings directly from environment variables without third-party dependencies:
- `RECOVEROPS_DB_PATH`: Path to persistent SQLite database. If unset, empty, or `:memory:`, a temporary ephemeral SQLite database is created and unlinked on shutdown.
- `RECOVEROPS_LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `INFO`.

---

## 9. Limitations & Phase 2 Roadmap

### Current Limitations
- **Synchronous Execution**: Tasks run in sequential order; concurrent parallel branch execution is scheduled for Phase 2.
- **Closed Action Vocabulary**: Remediation strategies are limited to `RETRY`, `RESET_SERVICE`, `RESTORE_FILE`, `SANITIZE_DATA`, and `ESCALATE`.
- **In-Memory Simulator**: Fault injection simulates network and hardware states within the local process space.

### Phase 2 Roadmap
- **RecoverBrain Integration**: Deep transformer advising strategy selection under strict deterministic oversight.
- **Vision Fault Telemetry**: Integrating multimodal sensor telemetry (e.g., optical/hardware inspection signals).
- **Pluggable Runners**: Secure worker adapters for external deterministic runtimes.

