import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from src.config import (
    TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
    N_CANDIDATES_PER_BATCH, MAX_ATTEMPTS_PER_SAMPLE
)
from src.hf_model import HFModel


SAMPLES_PATH = Path("data/samples_internal.json")
OUT_PATH = Path("data/verified_traces.json")

# Robustly extract final_answer even if JSON is malformed like {"final_answer":"B
FINAL_ANSWER_RE = re.compile(
    r'"final_answer"\s*:\s*"?([A-E])"?',
    re.IGNORECASE
)

WHITESPACE_RE = re.compile(r"\s+")


def build_prompt(sample: Dict[str, Any], mode: str) -> str:
    """
    mode:
      - "cot": encourages reasoning, but enforces JSON last line
      - "answer_only": NO reasoning, stricter output; often higher accuracy
    """
    q = sample["question"].strip()
    choices = sample["choices"]
    options_text = "\n".join([f"{k}) {v}" for k, v in choices.items()])

    if mode == "answer_only":
        return f"""You are a careful medical exam solver.
Choose exactly one option letter from the choices.

IMPORTANT OUTPUT RULE:
Return ONLY one JSON object on the last line, nothing else:
{{"final_answer":"A"}} (or B/C/D/E)

Question:
{q}

Choices:
{options_text}

Answer:
"""

    # default: "cot"
    return f"""You are a careful medical exam solver.

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


def extract_final_answer_letter(text: str) -> Optional[str]:
    """
    Extract final_answer letter without requiring valid JSON.
    Works even when output JSON is malformed/truncated.
    """
    m = None
    for m in FINAL_ANSWER_RE.finditer(text):
        pass
    if not m:
        return None
    return m.group(1).strip().upper()


def is_verified(output_text: str, correct_letter: str, valid_letters: List[str]) -> Tuple[bool, Optional[str]]:
    pred = extract_final_answer_letter(output_text)
    if pred is None:
        return False, None
    if pred not in valid_letters:
        return False, pred
    return pred == correct_letter, pred


def decoding_schedule(attempt: int) -> Tuple[str, float, float, int, bool]:
    """
    Returns (mode, temperature, top_p, n, debug_raw)
    Strategy:
      - start with your defaults
      - mid: lower temp
      - late: answer_only + very low temp (almost greedy)
    """
    if attempt <= 8:
        return ("cot", TEMPERATURE, TOP_P, N_CANDIDATES_PER_BATCH, True)
    if attempt <= 15:
        return ("cot", 0.4, max(TOP_P, 0.9), N_CANDIDATES_PER_BATCH, False)
    if attempt <= 25:
        return ("answer_only", 0.2, 0.95, N_CANDIDATES_PER_BATCH, False)
    # final rescue: near-greedy, answer_only, increase batch a bit
    return ("answer_only", 0.05, 1.0, max(N_CANDIDATES_PER_BATCH, 4), False)


def main():
    with open(SAMPLES_PATH, "r") as f:
        samples = json.load(f)

    model = HFModel()
    results = []

    for sample in samples:
        sample_id = sample["id"]
        correct = sample["correct_answer"].strip().upper()
        valid_letters = sorted(sample["choices"].keys())

        if correct not in valid_letters:
            raise ValueError(f"{sample_id}: correct_answer={correct} not in choices={valid_letters}")

        verified = []
        seen_hashes = set()
        attempts = 0
        pred_counter = Counter()

        while len(verified) < 3 and attempts < MAX_ATTEMPTS_PER_SAMPLE:
            attempts += 1
            mode, temp, top_p, n, debug_raw = decoding_schedule(attempts)
            prompt = build_prompt(sample, mode)

            print(f"[{sample_id}] attempt {attempts} (verified={len(verified)}) mode={mode} temp={temp} top_p={top_p} n={n}")

            outs = model.generate(
                prompt,
                n=n,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            for out in outs:
                if debug_raw and attempts <= 2:
                    print("\n--- RAW MODEL OUTPUT START ---")
                    print(out)
                    print("--- RAW MODEL OUTPUT END ---\n")

                # de-dup
                norm = WHITESPACE_RE.sub(" ", out).strip()
                h = hash(norm)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                ok, pred = is_verified(out, correct, valid_letters)
                if pred:
                    pred_counter[pred] += 1

                if ok:
                    verified.append({"text": out, "final_answer": correct})
                    if len(verified) >= 3:
                        break

            # small “signal” during long failures
            if attempts in (10, 20, 30) and len(verified) < 3:
                common = pred_counter.most_common(3)
                if common:
                    print(f"[{sample_id}] top preds so far: {common} (correct={correct})")

        if len(verified) < 3:
            # IMPORTANT CHANGE: don't crash the whole run; save partial + diagnostics
            results.append({
                "id": sample_id,
                "question": sample["question"],
                "choices": sample["choices"],
                "correct_answer": correct,
                "verified_traces": verified,
                "status": "FAILED_TO_VERIFY_3",
                "attempts": attempts,
                "pred_counts": dict(pred_counter),
            })
            print(f"[{sample_id}] ❌ only verified={len(verified)} after {attempts} attempts. continuing…")
            continue

        results.append({
            "id": sample_id,
            "question": sample["question"],
            "choices": sample["choices"],
            "correct_answer": correct,
            "verified_traces": verified,
            "status": "OK",
            "attempts": attempts,
        })

        print(f"[{sample_id}] ✅ verified=3 in attempts={attempts}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
