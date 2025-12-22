from abc import ABC, abstractmethod
from typing import List


class ModelInterface(ABC):
    """
    Abstract interface for all models used in the RLHF pipeline.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """
        Generate n completions for a given prompt.
        Returns a list of strings (raw model outputs).
        """
        pass

    @abstractmethod
    def logprob(self, prompt: str, completion: str) -> float:
        """
        Compute the total log-probability of `completion`
        conditioned on `prompt`.

        IMPORTANT:
        - Only log-probs of completion tokens
        - Prompt tokens must NOT be counted
        """
        pass
