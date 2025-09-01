"""
GGUF Key-Value Store Specification and Validation Rules.
This file acts as a Python-based rule registry for the GGUF format.
"""

from __future__ import annotations

from typing import Any, Dict

# This registry defines the expected type and constraints for known GGUF keys.
# The analyzer will use this to perform spec-conformance validation.
KNOWN_KEYS: Dict[str, Dict[str, Any]] = {
    # General (Required or Recommended)
    "general.architecture": {"type": "String", "pattern": r"^[a-z0-9]+$", "required": True},
    "general.alignment": {"type": "UInt32", "min": 8, "multiple_of": 8, "default": 32},
    "general.quantization_version": {"type": "UInt32", "min": 1},
    "general.file_type": {"type": "UInt32"},
    "general.name": {"type": "String"},
    # LLaMA Family (enforced if general.architecture == "llama")
    "llama.context_length": {"type": "UInt32", "min": 1},
    "llama.embedding_length": {"type": "UInt32", "min": 1},
    "llama.block_count": {"type": "UInt32", "min": 1},
    "llama.feed_forward_length": {"type": "UInt32", "min": 1},
    "llama.attention.head_count": {"type": "UInt32", "min": 1},
    "llama.attention.head_count_kv": {"type": "UInt32", "min": 1},
    "llama.rope.dimension_count": {"type": "UInt32", "min": 1},
    "llama.attention.layer_norm_rms_epsilon": {"type": "Float32", "min": 1e-9, "max": 1e-2},
    "llama.rope.scale": {"type": "Float32", "min": 1e-9},
    "llama.expert_count": {"type": "UInt32", "min": 1},
    "llama.expert_used_count": {"type": "UInt32", "min": 1},
    # Tokenizer
    "tokenizer.ggml.model": {"type": "String"},
    "tokenizer.ggml.tokens": {"type": "Array"},
    "tokenizer.ggml.scores": {"type": "Array"},
    "tokenizer.ggml.merges": {"type": "Array"},
    "tokenizer.ggml.bos_token_id": {"type": "UInt32"},
    "tokenizer.ggml.eos_token_id": {"type": "UInt32"},
    "tokenizer.chat_template": {"type": "String"},
}
