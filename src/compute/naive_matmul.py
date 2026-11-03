import numpy as np


def naive_matmul(A, B, simulator=None):
    """
    Matrix multiplication using the naive approach.

    Each simulated GPU thread calculates one output element.

    Threads are grouped into warps when counting global
    memory transactions.
    """

    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")

    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions")

    rows = A.shape[0]
    inner = A.shape[1]
    columns = B.shape[1]

    C = np.zeros((rows, columns), dtype=np.float32)

    # Calculate the output elements.
    for i in range(rows):
        for j in range(columns):

            total = 0.0

            for k in range(inner):
                total += A[i, k] * B[k, j]

            C[i, j] = total

    # Simulate global memory accesses.
    if simulator is not None:

        total_outputs = rows * columns

        # Process output elements in groups of warp_size.
        for warp_start in range(
            0,
            total_outputs,
            simulator.warp_size
        ):

            warp_end = min(
                warp_start + simulator.warp_size,
                total_outputs
            )

            # Each simulated lane calculates one output element.
            output_indices = range(warp_start, warp_end)

            # Every multiplication step accesses A and B.
            for k in range(inner):

                addresses_A = []
                addresses_B = []

                for output_index in output_indices:

                    i = output_index // columns
                    j = output_index % columns

                    # Use strides so that non-contiguous array views
                    # are also handled correctly.
                    address_A = (
                        A.ctypes.data
                        + i * A.strides[0]
                        + k * A.strides[1]
                    )

                    address_B = (
                        B.ctypes.data
                        + k * B.strides[0]
                        + j * B.strides[1]
                    )

                    addresses_A.append(address_A)
                    addresses_B.append(address_B)

                # Count the collective accesses of this warp.
                simulator.record_global_accesses(addresses_A)
                simulator.record_global_accesses(addresses_B)

    return C