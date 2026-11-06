# RecoverBrain

> **Neural Strategy Advisor for RecoverOps AI**  
> *A 97.5M-parameter custom decoder transformer trained on the NVIDIA RTX 5050 Laptop GPU (Blackwell, sm_120)*

---

## 1. Honest Framing & Scope
**RecoverBrain is a narrow domain specialist, NOT an Artificial General Intelligence (AGI).**

While the long-term architectural aspiration of autonomous operations is general cognitive resilience, a 97.5-million-parameter transformer trained from scratch on synthetic state transitions possesses domain-specific reasoning strictly bounded to failure diagnosis and strategy selection. We state this plainly to maintain absolute engineering integrity:
- It does **not** possess general world knowledge or open-ended reasoning capabilities.
- It is a specialized neural policy advisor optimized for closed-vocabulary failure classification and remediation proposals.
- Its value lies in establishing a mathematically bounded, auditable, and extensible architectural foundation (data engine, curriculum, safety arbitrator, and hardware-accelerated training pipeline) that can scale as additional task families and multimodal inputs are introduced.

---

## 2. Safety Model: The AI Proposes, The Engine Decides

In mission-critical infrastructure, an AI model must **never** hold direct execution privileges. RecoverBrain adheres to a strict "Propose-vs-Decide" safety architecture:

```
[Task Failure Injected]
         |
         v
[Phase 1 Recovery Guardrails]
   (1. Unrecoverable?)
   (2. Non-idempotent?) ---------> Guardrail Triggered? ---> YES ---> Immediate ESCALATE
   (3. Max attempts?)                                                  (AI Never Consulted)
   (4. Max recoveries?)
   (5. Shared budget?)
         | NO (Safe to evaluate strategy)
         v
[RecoverBrain Advisor]
   - Max 256 input tokens
   - Greedy deterministic decoding
   - Max 24 output tokens
   - Closed Vocabulary: RETRY | RESET_SERVICE | RESTORE_FILE | SANITIZE_DATA | ESCALATE
         |
         v
[Safety Arbitration Rule]
   * Proposed == ESCALATE        ---> ACCEPT (Safety signal honored)
                                      (Unless --escalate-confidence-threshold set and prob < threshold)
   * Proposed == Deterministic   ---> ACCEPT (Neural consensus)
   * Proposed != Deterministic   ---> OVERRIDE (Deterministic policy enforced)
   * Garbage / Out-of-Vocab      ---> ESCALATE (Fail-safe default)
         |
         v
[Relational SQLite Audit]
   - Persisted in 'brain_proposals' table (Parameterized SQL)
         |
         v
[WorkflowExecutor Dispatches Remediation]
   (WorkflowExecutor is the ONLY recovery budget owner)
```

### Safety Invariants:
1. **Model has zero execution capabilities**: No shell access, no network access, no `eval()`, `exec()`, or dynamic imports.
2. **Phase 1 guardrails run first**: Non-idempotent operations, exhausted budgets, or fatal hardware faults escalate instantly without consulting the model.
3. **Escalation policy**: If the model proposes `ESCALATE`, the engine respects it as a cautionary signal unless an optional confidence threshold is configured.
4. **Budget integrity**: The model cannot bypass, increment, or reset recovery budgets.

---

## 3. Diagnosis and Resolution of Class Collapse

In early experimental iterations, RecoverBrain collapsed to predicting `ESCALATE` for all inputs:
- **Root Cause**: Datagen initially generated non-idempotent tasks, exhausted budgets, and fatal hardware faults, allowing Phase 1 guardrails to trigger escalations in ~86% of training samples. Furthermore, cross-entropy loss was calculated over all tokens including the observation prefix. As a consequence, predicting `ESCALATE` trivially achieved 86% training accuracy while learning zero recovery behaviors.
- **Architectural Resolution**:
  1. **Strict Guardrail Exclusion**: Datagen now generates only instances where all Phase 1 guardrails pass, ensuring the model is trained exclusively on genuine strategy selection.
  2. **Class Stratification**: Exactly 20.00% balance across all 5 classes (`RETRY`, `RESET_SERVICE`, `RESTORE_FILE`, `SANITIZE_DATA`, `ESCALATE`).
  3. **Target-Only Loss (`--loss-on-target-only`)**: The observation prefix is masked with label `-100`, directing 100% of gradient descent into strategy selection and rationale generation.
  4. **Validation Macro-F1 Checkpointing**: Checkpointing and early stopping are governed by Validation Macro-F1 rather than cross-entropy loss.
  5. **Disjoint Novel Generalization Split**: The novel split is constructed from held-out parameter combinations that are verified to be 100% disjoint from training data.

---

## 4. Empirical Evaluation Metrics (Post-Fix)

Evaluated across the full 2,500-sample test, validation, and novel splits (7,500 samples per preset):

### Preset `base` (97,536,768 parameters):
| Metric | Validation Split (Full 2,500) | Test Split (Full 2,500) | Novel OOD Split (Full 2,500) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **100.00%** | **100.00%** | **95.68%** |
| **Majority-Class Baseline** | 20.00% | 20.00% | 20.00% |
| **Beats Majority Baseline?** | **YES (+80.00%)** | **YES (+80.00%)** | **YES (+75.68%)** |
| **Macro-F1** | **100.00%** | **100.00%** | **95.58%** |
| **Balanced Accuracy** | **100.00%** | **100.00%** | **95.68%** |
| **Disagreement Rate vs Policy** | **0.00%** | **0.00%** | **4.32%** |
| **Unnecessary ESCALATE Rate** | **0.00% (0 / 2,000)** | **0.00% (0 / 2,000)** | **0.00% (0 / 2,000)** |
| **Inference Throughput** | 30.5 tokens/s | 29.7 tokens/s | 31.0 tokens/s |

### Preset `tiny` (1,967,232 parameters):
| Metric | Validation Split (Full 2,500) | Test Split (Full 2,500) | Novel OOD Split (Full 2,500) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **100.00%** | **100.00%** | **100.00%** |
| **Majority-Class Baseline** | 20.00% | 20.00% | 20.00% |
| **Beats Majority Baseline?** | **YES (+80.00%)** | **YES (+80.00%)** | **YES (+80.00%)** |
| **Macro-F1** | **100.00%** | **100.00%** | **100.00%** |
| **Balanced Accuracy** | **100.00%** | **100.00%** | **100.00%** |
| **Disagreement Rate vs Policy** | **0.00%** | **0.00%** | **0.00%** |
| **Unnecessary ESCALATE Rate** | **0.00% (0 / 2,000)** | **0.00% (0 / 2,000)** | **0.00% (0 / 2,000)** |
| **Inference Throughput** | 87.5 tokens/s | 81.3 tokens/s | 78.1 tokens/s |

### Workflow-Level Benchmark (200 Seeded Workflows):
- **Baseline Deterministic Success Rate**: 75.00% (150 / 200)
- **Advisor-Enabled Success Rate**: 75.00% (150 / 200)
- **Workflow Status Agreement Rate**: **100.00%** (200 / 200)

---

## 5. CLI Commands

```bash
# 1. GPU Pre-flight Check
python -m brain.check_gpu

# 2. Balanced Synthetic Curriculum Data Generation (30k episodes)
python -m brain.datagen --episodes 30000

# 3. Train Tiny Preset (~2M params)
python -m brain.train --preset tiny --epochs 2

# 4. Train Base Preset (97.5M params)
python -m brain.train --preset base --epochs 2 --batch-size 16 --grad-accum 4

# 5. Full Evaluation (All 2,500 samples per split)
python -m brain.evaluate --preset base

# 6. Run Advised Engine Demonstration & 200-Workflow Benchmark
python -m brain.demo --preset base
```
