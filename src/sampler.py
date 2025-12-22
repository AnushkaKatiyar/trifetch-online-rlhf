import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.config import TEMPERATURE, TOP_P, MAX_NEW_TOKENS, N_CANDIDATES_PER_BATCH, MAX_ATTEMPTS_PER_SAMPLE
from src.hf_model import HFModel


SAMPLES_PATH = Path("data/samples_internal.json")
OUT_PATH = Path("data/verified_traces.json")

FINAL_JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)


def build_prompt(sample: Dict[str, Any]) -> str:
    """Strict prompt: force JSON-only final answer."""
    q = sample["question"].strip()
    choices = sample["choices"]

    # Render options in a stable format
    options_text = "\n".join([f"{k}) {v}" for k, v in choices.items()])

    prompt = f"""You are a careful medical exam solver.

Task:
1) Think step-by-step.
2) Choose exactly one option letter from the choices.
3) End with EXACTLY one JSON object on the last line:
{{"final_answer":"A"}}  (or B/C/D/E)

Question:
{q}

Choices:
{options_text}

Remember:
- The last line must be ONLY the JSON object.
- The JSON must contain only the key "final_answer".
- Do not include any extra text after the JSON.

Answer:
"""
    return prompt


def extract_last_json(text: str) -> Dict[str, Any]:
    """
    Extract the final JSON object from the end of the model output.
    We require it to be at the end to avoid ambiguous parsing.
    """
    m = FINAL_JSON_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON object found at end of output.")
    return json.loads(m.group(0))


def is_verified(output_text: str, correct_letter: str, valid_letters: List[str]) -> bool:
    try:
        obj = extract_last_json(output_text)
    except Exception:
        return False

    if "final_answer" not in obj or len(obj) != 1:
        return False

    pred = str(obj["final_answer"]).strip().upper()
    if pred not in valid_letters:
        return False

    return pred == correct_letter.strip().upper()


def main():
    with open(SAMPLES_PATH, "r") as f:
        samples = json.load(f)

    model = HFModel()
    results = []

    for sample in samples:
        sample_id = sample["id"]
        correct = sample["correct_answer"].strip().upper()
        valid_letters = sorted(sample["choices"].keys())  # e.g., ["A","B","C","D","E"]

        prompt = build_prompt(sample)

        verified = []
        seen_hashes = set()
        attempts = 0

        while len(verified) < 3 and attempts < MAX_ATTEMPTS_PER_SAMPLE:
            attempts += 1
            print(f"[{sample_id}] attempt {attempts} (verified={len(verified)})")


            # Generate a small batch each attempt for speed
            outs = model.generate(
                prompt,
                n=N_CANDIDATES_PER_BATCH,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            for out in outs:

                # DEBUG: inspect raw model output (limit to early attempts)
                if sample_id == "sample_1" and attempts <= 2:
                    print("\n--- RAW MODEL OUTPUT START ---")
                    print(out)
                    print("--- RAW MODEL OUTPUT END ---\n")

                # Make distinctness based on full text (including reasoning),
                # but normalize whitespace a bit to avoid trivial duplicates.
                norm = re.sub(r"\s+", " ", out).strip()
                h = hash(norm)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                if is_verified(out, correct, valid_letters):
                    verified.append({
                        "text": out,
                        "final_answer": correct
                    })
                    if len(verified) >= 3:
                        break


        if len(verified) < 3:
            # Fail loudly: this is better than silently producing incomplete data.
            raise RuntimeError(
                f"Could not get 3 verified traces for {sample_id}. "
                f"Got {len(verified)} after {attempts} attempts."
            )

        results.append({
            "id": sample_id,
            "question": sample["question"],
            "choices": sample["choices"],
            "correct_answer": correct,
            "verified_traces": verified
        })

        print(f"[{sample_id}] ✅ verified=3 in attempts={attempts}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
