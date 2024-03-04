import os





ROOT_DIRECTORY = f"{os.path.dirname(os.path.realpath(__file__))}/workspace"

MODELS_PATH = f"{ROOT_DIRECTORY}/models"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8


# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 8192 #4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = 8#100  # Llama-2-70B has 83 layers
N_BATCH = 512

### From experimenting with the Llama-2-7B-Chat-GGML model on 8GB VRAM, these values work:
# N_GPU_LAYERS = 20
# N_BATCH = 512


MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_ID_GPTQ = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
MODEL_BASENAME_GPTQ = "mistral-7b-instruct-v0.1.Q4_K_M.gptq"

