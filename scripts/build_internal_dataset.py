import json
import re
from pathlib import Path

RAW_PATH = Path("data/samples.json")
OUT_PATH = Path("data/samples_internal.json")


def parse_question_block(text: str):
    """
    Splits question text from multiple-choice options.
    Returns: question_str, choices_dict
    """
    lines = text.strip().splitlines()

    question_lines = []
    choices = {}

    option_pattern = re.compile(r"^([A-E])\)\s*(.+)$")

    for line in lines:
        match = option_pattern.match(line.strip())
        if match:
            letter, choice = match.groups()
            choices[letter] = choice.strip()
        else:
            question_lines.append(line)

    question = " ".join(question_lines).strip()
    return question, choices


def main():
    with open(RAW_PATH, "r") as f:
        raw_samples = json.load(f)

    internal_samples = []

    for idx, sample in enumerate(raw_samples, start=1):
        question_text = sample["Questions"]
        correct = sample["Answer"].strip()

        question, choices = parse_question_block(question_text)

        internal_samples.append({
            "id": f"sample_{idx}",
            "question": question,
            "choices": choices,
            "correct_answer": correct
        })

    with open(OUT_PATH, "w") as f:
        json.dump(internal_samples, f, indent=2)

    print(f"✅ Wrote {len(internal_samples)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
