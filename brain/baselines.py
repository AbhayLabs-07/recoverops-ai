"""Baseline models for strategy recommendation: Majority Class, Stateless Lookup, and Deterministic Decision Tree.

Implemented in pure Python stdlib and numpy only.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


CATEGORIES = [
    "timeout_error",
    "transient_service_error",
    "missing_file_error",
    "invalid_data_error",
    "rate_limit_exceeded",
    "transaction_deadlock",
    "disk_quota_exceeded",
    "flaky_service_degraded",
    "partially_corrupted_data",
    "cascaded_upstream_failure",
    "verification_failure",
    "unrecoverable_error",
    "none",
]

STRATEGIES = [
    "ESCALATE",
    "RESET_SERVICE",
    "RESTORE_FILE",
    "RETRY",
    "SANITIZE_DATA",
    "ROLLBACK_REPLAY",
    "PURGE_TEMP",
]


class MajorityClassBaseline:
    """Predicts the single most frequent target class observed during training."""

    def __init__(self) -> None:
        self.majority_class: str = "ESCALATE"

    def fit(self, targets: list[str]) -> None:
        if targets:
            counts = Counter(targets)
            self.majority_class = counts.most_common(1)[0][0]

    def predict(self, episode: dict[str, Any]) -> str:
        return self.majority_class


class StatelessLookupBaseline:
    """Stateless lookup baseline replicating the static Phase 1 mapping on failure category."""

    MAPPING = {
        "timeout_error": "RETRY",
        "transient_service_error": "RESET_SERVICE",
        "missing_file_error": "RESTORE_FILE",
        "invalid_data_error": "SANITIZE_DATA",
        "rate_limit_exceeded": "RETRY",
        "transaction_deadlock": "ROLLBACK_REPLAY",
        "disk_quota_exceeded": "PURGE_TEMP",
        "flaky_service_degraded": "RESET_SERVICE",
        "partially_corrupted_data": "SANITIZE_DATA",
        "cascaded_upstream_failure": "ESCALATE",
        "verification_failure": "ESCALATE",
        "unrecoverable_error": "ESCALATE",
        "none": "ESCALATE",
    }

    def predict(self, episode: dict[str, Any]) -> str:
        obs = episode.get("observation", {})
        if not isinstance(obs, dict) and "prompt" in episode:
            # Extract from prompt JSON string if needed
            p = episode["prompt"]
            try:
                obs_json = p.split("OBSERVATION:")[-1].split("STRATEGY:")[0].strip()
                obs = json.loads(obs_json)
            except Exception:
                obs = {}

        category = str(obs.get("failure_category", "none")).lower()
        return self.MAPPING.get(category, "ESCALATE")


def extract_features(episode: dict[str, Any]) -> np.ndarray:
    """Extract numeric and one-hot feature vector from an episode dictionary."""
    obs = episode.get("observation", {})
    if not isinstance(obs, dict) and "prompt" in episode:
        try:
            obs_json = episode["prompt"].split("OBSERVATION:")[-1].split("STRATEGY:")[0].strip()
            obs = json.loads(obs_json)
        except Exception:
            obs = {}

    cat = str(obs.get("failure_category", "none")).lower()
    cat_feats = [1.0 if cat == c else 0.0 for c in CATEGORIES]

    attempt = float(obs.get("attempt", 1))
    rec_used = float(obs.get("task_recoveries_used", 0))
    budget_left = float(obs.get("workflow_budget_left", 3))

    # History features
    history = obs.get("history", [])
    has_hist = 1.0 if len(history) > 0 else 0.0
    hist_len = float(len(history))
    last_strat = history[-1].get("strategy", "NONE") if history else "NONE"
    last_strat_feats = [1.0 if last_strat == s else 0.0 for s in STRATEGIES]
    same_failure_recurred = 1.0 if obs.get("same_failure_recurred", False) else 0.0

    # Ambiguity / parameters
    has_sanitized = 1.0 if obs.get("sanitized_data_configured", False) else 0.0
    has_transient = 1.0 if obs.get("transient_configured", False) else 0.0

    numeric = [attempt, rec_used, budget_left, has_hist, hist_len, same_failure_recurred, has_sanitized, has_transient]
    return np.array(cat_feats + last_strat_feats + numeric, dtype=np.float32)


class DecisionTreeNode:
    """Node in deterministic binary classification / regression tree."""

    def __init__(
        self,
        feature_idx: int = -1,
        threshold: float = 0.0,
        left: DecisionTreeNode | None = None,
        right: DecisionTreeNode | None = None,
        prediction: str = "ESCALATE",
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class DecisionTreeBaseline:
    """Small deterministic decision tree baseline trained via Gini impurity splits in pure numpy."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: DecisionTreeNode | None = None

    def fit(self, episodes: list[dict[str, Any]], targets: list[str]) -> None:
        if not episodes or not targets:
            return
        X = np.stack([extract_features(e) for e in episodes])
        y = np.array(targets)
        self.root = self._build_tree(X, y, depth=0)

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        most_common = Counter(y).most_common(1)[0][0]
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return DecisionTreeNode(prediction=most_common)

        best_gini = float("inf")
        best_feat = -1
        best_thresh = 0.0

        current_gini = self._gini(y)

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            # Sample candidate thresholds
            if len(vals) > 10:
                thresholds = np.percentile(vals, np.linspace(10, 90, 8))
            else:
                thresholds = (vals[:-1] + vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                w_gini = (np.sum(left_mask) * g_left + np.sum(right_mask) * g_right) / n_samples
                if w_gini < best_gini:
                    best_gini = w_gini
                    best_feat = feat
                    best_thresh = float(thresh)

        if best_gini >= current_gini or best_feat == -1:
            return DecisionTreeNode(prediction=most_common)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            prediction=most_common,
        )

    def predict(self, episode: dict[str, Any]) -> str:
        if self.root is None:
            return "ESCALATE"
        x = extract_features(episode)
        node = self.root
        while node.left is not None and node.right is not None:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction


class LogisticRegressionBaseline:
    """Multiclass Logistic Regression (Softmax) baseline in pure NumPy with L2 regularization."""

    def __init__(self, lr: float = 0.05, epochs: int = 250, l2: float = 1e-4) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None  # (n_features, n_classes)
        self.bias: np.ndarray | None = None     # (n_classes,)
        self.classes: list[str] = list(STRATEGIES)

    def fit(self, episodes: list[dict[str, Any]], targets: list[str]) -> None:
        if not episodes or not targets:
            return
        X = np.stack([extract_features(e) for e in episodes])
        n_samples, n_features = X.shape

        # Standardize features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std

        # Map targets to class indices
        self.classes = sorted(list(set(targets)))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        K = len(self.classes)

        # One-hot encoded targets
        Y = np.zeros((n_samples, K), dtype=np.float32)
        for i, t in enumerate(targets):
            if t in class_to_idx:
                Y[i, class_to_idx[t]] = 1.0

        # Initialize weights
        rng = np.random.RandomState(42)
        self.weights = rng.normal(0, 0.01, (n_features, K)).astype(np.float32)
        self.bias = np.zeros(K, dtype=np.float32)

        # Gradient Descent with Momentum
        v_w = np.zeros_like(self.weights)
        v_b = np.zeros_like(self.bias)
        momentum = 0.9

        for epoch in range(self.epochs):
            # Forward: logits = X W + b
            logits = np.dot(X_norm, self.weights) + self.bias
            # Softmax with numerical stability
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradient: (Probs - Y) / N
            grad = (probs - Y) / n_samples
            grad_w = np.dot(X_norm.T, grad) + self.l2 * self.weights
            grad_b = np.sum(grad, axis=0)

            v_w = momentum * v_w + self.lr * grad_w
            v_b = momentum * v_b + self.lr * grad_b
            self.weights -= v_w
            self.bias -= v_b

    def predict(self, episode: dict[str, Any]) -> str:
        if self.weights is None or self.bias is None:
            return "ESCALATE"
        x = extract_features(episode)
        x_norm = (x - self.mean) / self.std
        logits = np.dot(x_norm, self.weights) + self.bias
        idx = int(np.argmax(logits))
        return self.classes[idx]

    def predict_proba(self, episode: dict[str, Any]) -> tuple[str, float]:
        if self.weights is None or self.bias is None:
            return "ESCALATE", 1.0
        x = extract_features(episode)
        x_norm = (x - self.mean) / self.std
        logits = np.dot(x_norm, self.weights) + self.bias
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


def evaluate_baseline_predictions(
    predictions: list[str],
    targets: list[str],
    episodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate accuracy, macro-F1, balanced accuracy, and slice metrics."""
    assert len(predictions) == len(targets), "Predictions and targets must match in length"
    n = len(targets)
    if n == 0:
        return {}

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    acc = correct / n

    # Unique classes
    classes = sorted(list(set(targets)))
    f1s = []
    recalls = []

    for c in classes:
        tp = sum(1 for p, t in zip(predictions, targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(predictions, targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(predictions, targets) if p != c and t == c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        recalls.append(r)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    res = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "total_samples": n,
    }

    # Slice evaluations if episodes provided
    if episodes:
        slices = {
            "history_dependent": [i for i, e in enumerate(episodes) if e.get("is_history_dependent", False)],
            "ambiguous": [i for i, e in enumerate(episodes) if e.get("is_ambiguous", False)],
            "conflicting": [i for i, e in enumerate(episodes) if e.get("is_conflicting", False)],
        }
        slice_res = {}
        for s_name, indices in slices.items():
            if indices:
                s_preds = [predictions[i] for i in indices]
                s_targs = [targets[i] for i in indices]
                s_acc = sum(1 for p, t in zip(s_preds, s_targs) if p == t) / len(indices)
                slice_res[s_name] = {
                    "count": len(indices),
                    "accuracy": s_acc,
                }
        res["slices"] = slice_res

    return res

