"""Deterministic byte-level and domain vocabulary tokenizer for RecoverBrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brain.sandbox import resolve_brain_path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
UNK_ID = 4

# Domain keywords and subwords to assign dedicated single tokens
DOMAIN_VOCABULARY = [
    # Strategies & Outcomes
    "RETRY",
    "RESET_SERVICE",
    "RESTORE_FILE",
    "SANITIZE_DATA",
    "ESCALATE",
    "ROLLBACK_REPLAY",
    "PURGE_TEMP",
    "SUCCEEDED",
    "RECOVERED",
    "FAILED",
    "COMPLETED",
    "COMPLETED_WITH_RECOVERY",
    # Actions & Failure Categories
    "success",
    "transient_service",
    "timeout",
    "missing_file",
    "invalid_data",
    "unrecoverable",
    "rate_limited",
    "deadlock",
    "disk_quota",
    "transient_service_error",
    "timeout_error",
    "missing_file_error",
    "invalid_data_error",
    "verification_failure",
    "unrecoverable_error",
    "rate_limit_exceeded",
    "transaction_deadlock",
    "disk_quota_exceeded",
    "none",
    # Verification criteria
    "output_equals",
    "output_contains",
    "non_empty_output",
    "file_exists",
    "data_valid",
    "service_ready",
    # Structural keys
    "action",
    "error_code",
    "error_message",
    "attempt",
    "task_recoveries_used",
    "budget_left",
    "idempotent",
    "criterion_type",
    "target_strategy",
    "rationale",
    "observation",
    "proposal",
    "True",
    "False",
    "None",
    # Punctuation and formatting tokens
    ": ",
    ", ",
    '": "',
    '": ',
    '", "',
    '{"',
    '"}',
    '": {"',
    '": ["',
    '"], "',
    " -> ",
    "\n",
    "  ",
    "    ",
    "health",
    "online",
    "offline",
    "missing",
    "corrupt",
    "hardware",
    "exhausted",
    "budget",
    "idempotency",
    "transient",
    "permanent",
    "remediation",
    "verified",
]


class BrainTokenizer:
    """Deterministic tokenizer with byte-level fallback for guaranteed round-trip fidelity."""

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.byte_to_id: dict[int, int] = {}
        self.id_to_byte: dict[int, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Construct fixed vocabulary up to vocab_size."""
        current_id = 0

        # 1. Special tokens
        for s in SPECIAL_TOKENS:
            self.token_to_id[s] = current_id
            self.id_to_token[current_id] = s
            current_id += 1

        # 2. Byte tokens (0..255) to guarantee 100% round-trip fidelity for any UTF-8 text
        for b in range(256):
            b_token = f"<byte_{b}>"
            self.token_to_id[b_token] = current_id
            self.id_to_token[current_id] = b_token
            self.byte_to_id[b] = current_id
            self.id_to_byte[current_id] = b
            current_id += 1

        # 3. Domain vocabulary terms
        for term in DOMAIN_VOCABULARY:
            if current_id >= self.vocab_size:
                break
            if term not in self.token_to_id:
                self.token_to_id[term] = current_id
                self.id_to_token[current_id] = term
                current_id += 1

        # 4. Fill remaining vocabulary slots deterministically with dummy token IDs
        while current_id < self.vocab_size:
            dummy = f"<unused_{current_id}>"
            self.token_to_id[dummy] = current_id
            self.id_to_token[current_id] = dummy
            current_id += 1

        # Sort multi-character tokens descending by length for greedy matching
        self._multichar_tokens = sorted(
            [t for t in self.token_to_id.keys() if not t.startswith("<") and len(t) > 1],
            key=lambda x: len(x),
            reverse=True,
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs using greedy longest-token matching with byte fallback."""
        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(BOS_ID)

        i = 0
        n = len(text)
        while i < n:
            matched = False
            # Check multichar tokens
            for token_str in self._multichar_tokens:
                if text.startswith(token_str, i):
                    tokens.append(self.token_to_id[token_str])
                    i += len(token_str)
                    matched = True
                    break

            if not matched:
                # Single character check
                ch = text[i]
                if ch in self.token_to_id:
                    tokens.append(self.token_to_id[ch])
                    i += 1
                else:
                    # Fall back to UTF-8 bytes
                    raw_bytes = ch.encode("utf-8")
                    for b in raw_bytes:
                        tokens.append(self.byte_to_id[b])
                    i += 1

        if add_special_tokens:
            tokens.append(EOS_ID)

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a reconstructed string."""
        byte_buffer = bytearray()
        string_parts: list[str] = []

        def flush_bytes():
            if byte_buffer:
                string_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            if tid in (PAD_ID, BOS_ID, EOS_ID, SEP_ID):
                flush_bytes()
                continue
            if tid in self.id_to_byte:
                byte_buffer.append(self.id_to_byte[tid])
            elif tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if not tok.startswith("<"):
                    flush_bytes()
                    string_parts.append(tok)

        flush_bytes()
        return "".join(string_parts)

    def save(self, destination: str | Path = "data/tokenizer.json") -> None:
        """Persist tokenizer vocabulary into brain/sandbox/."""
        if isinstance(destination, Path):
            target = destination
        else:
            target = resolve_brain_path(destination)
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, relative_path: str) -> BrainTokenizer:
        """Load tokenizer vocabulary from brain/sandbox/."""
        source = resolve_brain_path(relative_path)
        data = json.loads(source.read_text(encoding="utf-8"))
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        return tok

