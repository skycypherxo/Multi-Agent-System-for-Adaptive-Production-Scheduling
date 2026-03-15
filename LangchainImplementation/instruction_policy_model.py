"""
Trainable instruction policy model for end-to-end scheduling decisions.

This model is intentionally lightweight and offline:
- text classifier (Multinomial Naive Bayes) for capability prediction
- learned planning priors (start offsets + preconditions) from dataset
- optional persisted artifact for faster startup
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence


DATASET_PATH = Path(__file__).with_name("data") / "instruction_policy_dataset.jsonl"
ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "instruction_policy_model.json"
_TOKEN_RE = re.compile(r"[a-z0-9_\-]+")


@dataclass
class TrainingExample:
    instruction: str
    capability: str
    rush: bool
    start_offset: int
    preferred_machine: str
    preconditions: List[str]


class InstructionPolicyModel:
    """Tiny trainable policy used by scheduler and machine agents."""

    def __init__(self, dataset_path: Path | None = None, artifact_path: Path | None = None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.artifact_path = artifact_path or ARTIFACT_PATH

        self.class_counts: Dict[str, int] = {}
        self.token_counts: Dict[str, Dict[str, int]] = {}
        self.total_tokens_per_class: Dict[str, int] = {}
        self.vocab: set[str] = set()

        self.preferred_machine_counts: Dict[str, Dict[str, int]] = {}
        self.start_offsets: Dict[str, Dict[str, float]] = {}
        self.preconditions: Dict[str, List[str]] = {}

        self.total_examples: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall((text or "").lower())

    def _load_examples(self) -> List[TrainingExample]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Instruction dataset not found: {self.dataset_path}")

        examples: List[TrainingExample] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(
                    TrainingExample(
                        instruction=str(row.get("instruction", "")).strip(),
                        capability=str(row.get("capability", "")).strip().lower(),
                        rush=bool(row.get("rush", False)),
                        start_offset=int(row.get("start_offset", 0)),
                        preferred_machine=str(row.get("preferred_machine", "")).strip(),
                        preconditions=list(row.get("preconditions", [])),
                    )
                )

        if not examples:
            raise ValueError("Instruction dataset is empty.")
        return examples

    def train(self) -> None:
        examples = self._load_examples()

        class_counts: Counter[str] = Counter()
        token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        total_tokens_per_class: Counter[str] = Counter()
        vocab: set[str] = set()

        preferred_machine_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        offset_accumulator: Dict[str, List[int]] = defaultdict(list)
        precondition_counts: Dict[str, Counter[str]] = defaultdict(Counter)

        for example in examples:
            capability = example.capability
            class_counts[capability] += 1

            tokens = self._tokenize(example.instruction)
            token_counts[capability].update(tokens)
            total_tokens_per_class[capability] += len(tokens)
            vocab.update(tokens)

            preferred_machine_counts[capability][example.preferred_machine] += 1

            rush_key = "rush" if example.rush else "normal"
            offset_accumulator[f"{capability}:{rush_key}"].append(example.start_offset)
            offset_accumulator[f"{capability}:all"].append(example.start_offset)

            precondition_counts[capability].update([p.strip().lower() for p in example.preconditions if p])

        self.class_counts = dict(class_counts)
        self.token_counts = {k: dict(v) for k, v in token_counts.items()}
        self.total_tokens_per_class = dict(total_tokens_per_class)
        self.vocab = vocab
        self.preferred_machine_counts = {k: dict(v) for k, v in preferred_machine_counts.items()}
        self.start_offsets = {
            key: {
                "avg": float(sum(vals) / max(1, len(vals))),
                "count": float(len(vals)),
            }
            for key, vals in offset_accumulator.items()
        }
        self.preconditions = {
            cap: [p for p, _ in precondition_counts[cap].most_common(5)]
            for cap in precondition_counts
        }
        self.total_examples = len(examples)

    def save(self) -> None:
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_counts": self.class_counts,
            "token_counts": self.token_counts,
            "total_tokens_per_class": self.total_tokens_per_class,
            "vocab": sorted(self.vocab),
            "preferred_machine_counts": self.preferred_machine_counts,
            "start_offsets": self.start_offsets,
            "preconditions": self.preconditions,
            "total_examples": self.total_examples,
        }
        self.artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.artifact_path.exists():
            return False

        payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.class_counts = {str(k): int(v) for k, v in payload.get("class_counts", {}).items()}
        self.token_counts = {
            str(k): {str(t): int(c) for t, c in v.items()}
            for k, v in payload.get("token_counts", {}).items()
        }
        self.total_tokens_per_class = {
            str(k): int(v) for k, v in payload.get("total_tokens_per_class", {}).items()
        }
        self.vocab = set(payload.get("vocab", []))
        self.preferred_machine_counts = {
            str(k): {str(m): int(c) for m, c in v.items()}
            for k, v in payload.get("preferred_machine_counts", {}).items()
        }
        self.start_offsets = {
            str(k): {str(sk): float(sv) for sk, sv in v.items()}
            for k, v in payload.get("start_offsets", {}).items()
        }
        self.preconditions = {
            str(k): [str(x) for x in v]
            for k, v in payload.get("preconditions", {}).items()
        }
        self.total_examples = int(payload.get("total_examples", 0))
        return True

    def ensure_ready(self) -> None:
        if self.load():
            return
        self.train()
        self.save()

    def predict_capability(self, instruction: str, fallback: str = "") -> str:
        if not self.class_counts:
            self.ensure_ready()

        classes = list(self.class_counts.keys())
        if not classes:
            return fallback

        tokens = self._tokenize(instruction)
        if not tokens and fallback:
            return fallback

        vocab_size = max(1, len(self.vocab))
        best_class = classes[0]
        best_score = float("-inf")

        for capability in classes:
            prior = (self.class_counts[capability] + 1.0) / (self.total_examples + len(classes))
            score = math.log(prior)
            denom = self.total_tokens_per_class.get(capability, 0) + vocab_size
            class_token_counts = self.token_counts.get(capability, {})
            for token in tokens:
                score += math.log((class_token_counts.get(token, 0) + 1.0) / denom)
            if score > best_score:
                best_score = score
                best_class = capability

        return best_class or fallback

    @staticmethod
    def _machine_capabilities(machine) -> List[str]:
        skills = getattr(machine, "skills", []) or []
        return [s.id for s in skills if s.id != "status_report"]

    def choose_machine(self, job: Dict, available_machines: Sequence, rag_context: str = "") -> Optional[str]:
        if not available_machines:
            return None

        instruction = (
            f"job type {job.get('job_type', '')} priority {job.get('priority', 'normal')} "
            f"duration {job.get('duration_minutes', 0)} {rag_context}"
        )
        fallback_capability = str(job.get("job_type", "")).strip().lower()
        predicted_capability = self.predict_capability(instruction, fallback=fallback_capability)

        capable = []
        for machine in available_machines:
            machine_caps = self._machine_capabilities(machine)
            if predicted_capability in machine_caps:
                capable.append(machine)

        # Fallback to job_type if classifier predicted something unavailable.
        if not capable and fallback_capability:
            for machine in available_machines:
                machine_caps = self._machine_capabilities(machine)
                if fallback_capability in machine_caps:
                    capable.append(machine)

        if not capable:
            capable = list(available_machines)

        preferred_counts = self.preferred_machine_counts.get(predicted_capability, {})
        capable.sort(
            key=lambda m: (
                -preferred_counts.get(getattr(m, "name", ""), 0),
                getattr(m, "name", ""),
            )
        )

        return getattr(capable[0], "name", None) if capable else None

    def plan_task(self, task_name: str, duration_minutes: int, priority: str = "normal") -> Dict[str, object]:
        capability = self.predict_capability(task_name, fallback=task_name.strip().lower())
        rush_key = "rush" if str(priority).lower() == "high" else "normal"

        offset_key = f"{capability}:{rush_key}"
        fallback_key = f"{capability}:all"
        avg_offset = self.start_offsets.get(offset_key, {}).get("avg")
        if avg_offset is None:
            avg_offset = self.start_offsets.get(fallback_key, {}).get("avg", 0.0)

        duration = int(duration_minutes)
        if capability == "painting":
            # Learned behavior: painting typically needs setup/drying buffer.
            duration += 5

        learned_preconditions = self.preconditions.get(capability, [])
        preconditions = learned_preconditions[:3] if learned_preconditions else ["machine_ready"]

        return {
            "plan_start_offset_minutes": int(round(avg_offset or 0.0)),
            "plan_duration_minutes": max(1, duration),
            "preconditions": preconditions,
        }


def build_or_load_instruction_policy() -> InstructionPolicyModel:
    model = InstructionPolicyModel()
    model.ensure_ready()
    return model


if __name__ == "__main__":
    model = build_or_load_instruction_policy()
    print("Instruction policy model ready")
    print(f"artifact: {model.artifact_path}")
    print(f"dataset: {model.dataset_path}")
    print(f"examples: {model.total_examples}")
    print(f"classes: {sorted(model.class_counts.keys())}")
