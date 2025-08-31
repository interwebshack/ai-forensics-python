# ai_forensics/model_formats/gguf/gguf_quantization.py
"""
GGUF Quantization Types (GGML) and metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class GGMLType(IntEnum):
    """GGML tensor types, including quantization."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # Deprecated
    # Q4_2 = 4
    # Q4_3 = 5
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 16
    I16 = 17
    I32 = 18
    I64 = 19
    F64 = 20
    # Special types
    COUNT = 21


@dataclass
class QuantizationInfo:
    """Properties of a GGML quantization type."""

    block_size: int
    bits_per_weight: float

    def get_expected_size(self, n_elements: int) -> int:
        """Calculate the expected byte size for a tensor with this quantization."""
        if self.block_size == 0 or self.bits_per_weight == 0:
            return 0  # Should not happen for quantized types

        # Calculate size based on blocks
        if self.block_size > 1:
            num_blocks = n_elements // self.block_size
            if n_elements % self.block_size != 0:
                # Should not happen with well-formed tensors
                return -1  # Indicate error

            # Example for Q4_0: 2 bytes for scale + (32 weights * 4 bits)/8 = 16 bytes. Total 18 bytes for 32 elements.
            # This is a simplification; a more precise formula depends on the exact format (e.g. scale sizes)
            # For this example, we'll use an approximation.
            bytes_per_block = (self.block_size * self.bits_per_weight) / 8.0
            # Many formats add scales (e.g., F16) per block
            if (
                "K" in GGMLType(self.bits_per_weight).name
                or "Q" in GGMLType(self.bits_per_weight).name
            ):
                bytes_per_block += 2  # Common size for scale factor
            return int(num_blocks * bytes_per_block)

        return int(n_elements * self.bits_per_weight / 8.0)


# Mapping from GGMLType to its properties
# Note: bits_per_weight can be fractional for complex types like K-quants
QUANTIZATION_MAP = {
    GGMLType.F32: QuantizationInfo(1, 32),
    GGMLType.F16: QuantizationInfo(1, 16),
    GGMLType.Q4_0: QuantizationInfo(32, 4.5),  # 4 bits + 16-bit scale per 32 = 4.5
    GGMLType.Q8_0: QuantizationInfo(32, 8.5),  # 8 bits + 16-bit scale per 32
    GGMLType.Q2_K: QuantizationInfo(256, 2.65625),  # Complex calculation for K-quants
    GGMLType.Q3_K: QuantizationInfo(256, 3.65625),
    GGMLType.Q4_K: QuantizationInfo(256, 4.65625),
    GGMLType.Q5_K: QuantizationInfo(256, 5.65625),
    GGMLType.Q6_K: QuantizationInfo(256, 6.78125),
    GGMLType.Q8_K: QuantizationInfo(256, 8.78125),
    # Add other types as needed
}
