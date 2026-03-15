"""Train and persist the instruction policy model from dataset."""

from instruction_policy_model import build_or_load_instruction_policy


def main() -> None:
    model = build_or_load_instruction_policy()
    print("Instruction policy trained/loaded successfully")
    print(f"Dataset: {model.dataset_path}")
    print(f"Artifact: {model.artifact_path}")
    print(f"Examples: {model.total_examples}")
    print(f"Classes: {sorted(model.class_counts.keys())}")


if __name__ == "__main__":
    main()
