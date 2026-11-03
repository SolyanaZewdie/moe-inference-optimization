import numpy as np


def quantize_blockwise(weights, bits=8, block_size=32):
    """
    Quantize weights using block-wise symmetric quantization.

    Returns:
        quantized: Integer representation of the weights.
        scales: One scale for each block.
    """

    weights = np.asarray(weights, dtype=np.float32)

    if bits not in (4, 8):
        raise ValueError("Only INT4 and INT8 are supported")

    if block_size <= 0:
        raise ValueError("Block size must be positive")

    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must contain only finite values")

    # Flatten the weights so we can process them in blocks.
    flat_weights = weights.flatten()

    # Maximum positive integer value for symmetric quantization.
    if bits == 8:
        max_value = 127
    else:
        max_value = 7

    quantized = np.zeros(flat_weights.shape, dtype=np.int8)

    # Calculate the number of blocks.
    num_blocks = (len(flat_weights) + block_size - 1) // block_size

    scales = np.zeros(num_blocks, dtype=np.float32)

    # Quantize each block independently.
    for block_index in range(num_blocks):

        start = block_index * block_size
        end = min(start + block_size, len(flat_weights))

        block = flat_weights[start:end]

        # Find the largest absolute value in this block.
        max_absolute = np.max(np.abs(block))

        # Avoid division by zero for blocks containing only zeros.
        if max_absolute == 0:
            scale = 1.0
        else:
            scale = max_absolute / max_value

        scales[block_index] = scale

        # Convert floating-point values into integers.
        quantized[start:end] = np.clip(
            np.round(block / scale),
            -max_value,
            max_value
        ).astype(np.int8)

    return quantized.reshape(weights.shape), scales


def dequantize_blockwise(quantized, scales, block_size):
    """
    Reconstruct approximate floating-point weights.
    """

    quantized = np.asarray(quantized)
    scales = np.asarray(scales)

    if block_size <= 0:
        raise ValueError("Block size must be positive")

    if not np.issubdtype(quantized.dtype, np.integer):
        raise ValueError("Quantized weights must be integers")

    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Scales must be finite and positive")

    flat_quantized = quantized.flatten()

    num_blocks = (len(flat_quantized) + block_size - 1) // block_size

    if len(scales) != num_blocks:
        raise ValueError("Incorrect number of scales")

    reconstructed = np.zeros(
        flat_quantized.shape,
        dtype=np.float32
    )

    # Reconstruct each block using its corresponding scale.
    for block_index in range(num_blocks):

        start = block_index * block_size
        end = min(start + block_size, len(flat_quantized))

        reconstructed[start:end] = (
            flat_quantized[start:end] * scales[block_index]
        )

    return reconstructed.reshape(quantized.shape)


def pack_int4(values):
    """
    Pack two signed INT4 values into one byte.

    Returns:
        packed: Packed unsigned bytes.
        original_size: Number of original INT4 values.
    """

    values = np.asarray(values)

    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("INT4 values must be integers")

    flat_values = values.flatten()

    if np.any(flat_values < -8) or np.any(flat_values > 7):
        raise ValueError("INT4 values must be between -8 and 7")

    original_size = len(flat_values)

    # Two values are stored in each byte.
    if original_size % 2 != 0:
        flat_values = np.append(flat_values, 0)

    # Convert signed values into their 4-bit representation.
    unsigned_values = flat_values.astype(np.int8) & 0x0F

    # First value occupies the lower four bits.
    # Second value occupies the upper four bits.
    low = unsigned_values[0::2]
    high = unsigned_values[1::2]

    packed = low | (high << 4)

    return packed.astype(np.uint8), original_size


def unpack_int4(packed, original_size):
    """
    Unpack bytes into signed INT4 values.
    """

    packed = np.asarray(packed, dtype=np.uint8)

    if original_size < 0 or original_size > len(packed) * 2:
        raise ValueError("Invalid original size")

    # Extract the lower and upper four-bit values.
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # Reconstruct the original sequence.
    values = np.empty(len(packed) * 2, dtype=np.int8)

    values[0::2] = low
    values[1::2] = high

    # Convert 4-bit two's complement back to signed integers.
    values[values >= 8] -= 16

    # Remove padding if the original number was odd.
    return values[:original_size]