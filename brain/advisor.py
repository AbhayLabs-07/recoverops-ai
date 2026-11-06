"""Inference advisor wrapping RecoverBrain under strict input/output bounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from brain.config import get_preset
from brain.evaluate import parse_model_proposal
from brain.model import RecoverBrainModel
from brain.sandbox import resolve_brain_path
from brain.tokenizer import BrainTokenizer, BOS_ID, SEP_ID, EOS_ID


class BrainAdvisor:
    """Inference advisor serving recommendations under strict sandboxing and bounded generation."""

    def __init__(
        self,
        preset_name: str = "base",
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.preset_name = preset_name
        self.config = get_preset(preset_name)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = RecoverBrainModel(self.config).to(self.device)

        # Load weights safely
        ckpt_dir = resolve_brain_path("checkpoints")
        if checkpoint_path is None:
            candidates = [
                ckpt_dir / f"checkpoint_{preset_name}_best.pt",
                ckpt_dir / f"checkpoint_{preset_name}_final.pt",
                ckpt_dir / f"checkpoint_{preset_name}_latest.pt",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found is None:
                raise FileNotFoundError(
                    f"No checkpoint available for preset '{preset_name}' in {ckpt_dir}."
                )
            target_ckpt = found
        else:
            target_ckpt = Path(checkpoint_path)

        from brain.config import ModelConfig
        try:
            torch.serialization.add_safe_globals([ModelConfig])
        except Exception:
            pass
        state = torch.load(str(target_ckpt), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        # Load tokenizer
        data_dir = resolve_brain_path("data")
        tok_file = data_dir / "tokenizer.json"
        if tok_file.exists():
            self.tokenizer = BrainTokenizer.load("data/tokenizer.json")
        else:
            self.tokenizer = BrainTokenizer(vocab_size=self.config.vocab_size)

    def propose_strategy(
        self,
        observation: dict[str, Any],
        vision_signal: dict[str, Any] | None = None,
    ) -> tuple[str, str, float]:
        """Generate a proposed recovery strategy from observation telemetry.

        Returns:
            tuple of (proposed_strategy: str, rationale: str, confidence: float)
        """
        obs_copy = dict(observation)
        if vision_signal is not None:
            # Documented extension point: incorporate optional visual telemetry
            obs_copy["vision_signal"] = vision_signal

        obs_json = json.dumps(obs_copy, sort_keys=True)
        prompt = f"OBSERVATION: {obs_json}\nSTRATEGY:"

        # Format prompt tokens aligned with training: [BOS_ID] + encode(prompt) + [SEP_ID]
        prompt_tokens = [BOS_ID] + self.tokenizer.encode(prompt)[:250] + [SEP_ID]

        with torch.no_grad():
            # Compute confidence on next token
            inp = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            logits, _ = self.model(inp)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            pred_token = int(torch.argmax(next_token_logits).item())
            confidence = float(probs[pred_token].item())

            # Autoregressive greedy generation
            gen_tokens = self.model.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=24,
                eos_token_id=EOS_ID,
            )

        new_tokens = gen_tokens[len(prompt_tokens) :]
        raw_output = self.tokenizer.decode(new_tokens)

        # Closed-vocabulary parsing; garbage -> ESCALATE
        strategy, rationale = parse_model_proposal(raw_output)
        return strategy, rationale, confidence
