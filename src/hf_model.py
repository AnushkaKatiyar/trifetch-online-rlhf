import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from src.model_interface import ModelInterface
from src.config import MODEL_NAME, DEVICE


class HFModel(ModelInterface):
    def __init__(self):
        """
        Hugging Face causal LM wrapper.

        - Supports CPU (local) and CUDA (Colab)
        - Supports models with custom HF code (e.g. HuatuoGPT)
        """

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=False,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Model ---
        if DEVICE == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """
        Sample n completions from the model.
        Returns ONLY the generated completion (prompt removed).
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        prompt_len = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        for seq in outputs:
            gen_ids = seq[prompt_len:]  # slice by token length (CRITICAL)
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text.strip())

        return completions

    @torch.no_grad()
    def logprob(self, prompt: str, completion: str) -> float:
        """
        Compute log P(completion | prompt).

        This is used by:
        - DPO (best vs worst comparison)
        - Reference model comparisons
        """

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.model.device)

        full_ids = self.tokenizer(
            prompt + completion,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.model.device)

        outputs = self.model(full_ids)
        logits = outputs.logits

        # Standard causal LM shift
        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]

        completion_start = prompt_ids.shape[1]

        completion_logits = shift_logits[:, completion_start - 1 :, :]
        completion_labels = shift_labels[:, completion_start - 1 :]

        log_probs = torch.nn.functional.log_softmax(
            completion_logits,
            dim=-1
        )

        token_logprobs = log_probs.gather(
            2,
            completion_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_logprobs.sum().item()


if __name__ == "__main__":
    # Quick sanity test (local CPU only)
    model = HFModel()

    prompt = (
        "Answer with A, B, C, or D.\n"
        "What is 2+2?\n"
        "A) 3\nB) 4\nC) 5\nD) 6\n\n"
        "FINAL ANSWER:\n"
    )
    completion = '{"final_answer":"B"}'

    lp = model.logprob(prompt, completion)
    print("Log-prob:", lp)
