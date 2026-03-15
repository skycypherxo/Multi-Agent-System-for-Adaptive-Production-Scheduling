from knowledge_base import SCHEDULING_KNOWLEDGE
from micro_language_model import MicroByteLM


def main() -> None:
    model = MicroByteLM(embed_dim=4, seed=0)
    print(f"MicroByteLM parameters: {model.num_parameters()}")
    loss = model.fit(SCHEDULING_KNOWLEDGE, steps=1500, lr=0.15, batch_size=64)
    print(f"Final batch NLL: {loss:.4f}")
    print()
    text = model.generate("Scheduling rule: ", max_new_bytes=200, temperature=0.9)
    # Make console-printable on Windows shells that default to cp1252.
    print(text.encode("cp1252", errors="replace").decode("cp1252", errors="replace"))


if __name__ == "__main__":
    main()
