MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" 

TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 200

N_CANDIDATES_PER_BATCH = 1      # keep small on CPU
MAX_ATTEMPTS_PER_SAMPLE = 40    # strict; fail loud if not reached

