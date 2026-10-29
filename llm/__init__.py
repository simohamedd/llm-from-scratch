"""llm-from-scratch: build, train and serve a small modern LLM in plain PyTorch."""
from .config import PRESETS, ModelConfig
from .model import GPT, KVCache
from .tokenizer import BPETokenizer
