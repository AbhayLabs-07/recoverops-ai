# RecoverOps AI — Phase 1 Core Engine

RecoverOps AI is a small, deterministic self-healing workflow engine for the
OmniRush Buildathon. It executes a sequence of safe built-in tasks, diagnoses
known failures, applies bounded recovery strategies, and independently checks
the result before reporting success.

Phase 1 intentionally does **not** include a dashboard, an LLM, external
services, arbitrary shell commands, dynamic code execution, or untrusted
instruction handling.

## Requirements

- Python 3.11 (the target version is pinned in `.python-version`)
- SQLite, provided by Python's standard library
- No third-party runtime packages; `requirements.txt` is the reproducible
  zero-dependency lockfile

## Setup

```bash
python --version
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --requirement requirements.txt
```

There are no required secrets or external services. Copy `.env.example` only
if a local environment file is useful; Phase 1 currently reads no environment
variables.

## Run the demonstration

```bash
python -m app.main
# or
make run
```

The CLI runs three workflows and prints a readable timeline:

1. A normal workflow that executes and verifies successfully.
2. A transient service failure that is reset, retried, and verified.
3. An unrecoverable failure that terminates safely with human escalation.

The demonstration database is created in a temporary directory and removed on
exit. Applications can use `Storage("recoverops.db")` for a persistent local
audit database; local database files are excluded by `.gitignore`.

## Run tests

```bash
python -m unittest discover -s tests -v
# or
make test
```

The suite covers successful execution, transient failures, timeouts, missing
files, invalid data, verification failures, retry and workflow budgets,
unrecoverable errors, non-idempotent task escalation, invalid configuration,
and SQLite persistence.

## Architecture

```text
WorkflowExecutor
  ├── FailureSimulator       closed set of deterministic task actions
  ├── RecoveryManager        failure classification and safe strategy choice
  ├── Verifier               explicit, independent success checks
  └── Storage                parameterized SQLite audit persistence
```

### Models (`app/models.py`)

Typed dataclasses and enums model tasks, workflows, direct execution results,
verification results, recovery attempts, task outcomes, and final workflow
states. A task declares its action, success criteria, idempotence, maximum
attempts, and maximum recoveries. Workflow construction rejects empty or
duplicate task IDs and invalid budgets.

### Failure simulation (`app/failure_simulator.py`)

Only these actions are available in Phase 1:

- `success`
- `transient_service`
- `timeout`
- `missing_file`
- `invalid_data`
- `unrecoverable`

Failure timing is deterministic from the task parameters and attempt number.
File operations are constrained to an optional simulator root directory.

### Verification (`app/verification.py`)

The verifier checks declared criteria such as output equality, output
containment, non-empty output, file existence, data validity, and service
readiness. An action returning without an exception is never sufficient for
success; a failed verification always leaves the task and workflow in a
non-success state unless a later independently verified retry passes.

### Recovery (`app/recovery.py`)

Known failures map only to predefined strategies: retry, reset service, restore
an explicitly configured file, sanitize known simulated data, or escalate.
Retries are bounded by task attempts, task recovery limits, and a shared
workflow recovery budget. Non-idempotent tasks are escalated instead of being
repeated blindly. Every applied recovery or escalation is recorded.

### Storage (`app/storage.py`)

SQLite stores the workflow parent row, every direct task execution, every
recovery attempt, and every verification result. Writes use parameterized SQL
and transaction context managers. Database files remain local and are ignored
by Git.

## Using the engine in Python

```python
from app.failure_simulator import FailureSimulator
from app.models import CriterionType, SuccessCriteria, Task, TaskAction, Workflow
from app.verification import Verifier
from app.workflow import WorkflowExecutor

task = Task(
    id="health-check",
    name="Health check",
    action=TaskAction.SUCCESS,
    parameters={"output": {"healthy": True}},
    success_criteria=SuccessCriteria(
        CriterionType.OUTPUT_EQUALS, {"healthy": True}
    ),
)
workflow = Workflow(name="example", tasks=(task,))
result = WorkflowExecutor(
    simulator=FailureSimulator(),
    verifier=Verifier(),
).execute(workflow)
assert result.succeeded
```

## Docker

The included Dockerfile uses Python 3.11, installs the locked dependency file,
and runs the CLI without system-wide project dependencies:

```bash
docker build -t recoverops-ai .
docker run --rm recoverops-ai
```

For a local test command in the image:

```bash
docker run --rm recoverops-ai python -m unittest discover -s tests -v
```

## Limitations and next phase boundary

Phase 1 is a reliable local engine and deterministic demonstration, not a
production orchestrator. It has no process isolation, distributed execution,
authentication, remote service adapters, dashboard, LLM integration, durable
job queue, or advanced backoff scheduling. Those are intentionally outside
this phase and should not be assumed from the simulator actions.
