import numpy as np


def record_tile_accesses(
    array,
    row_start,
    column_start,
    tile_rows,
    tile_columns,
    simulator
):
    """
    Simulate global memory transactions when loading a tile.

    Each warp loads up to warp_size elements.
    """

    addresses = []

    for i in range(tile_rows):
        for j in range(tile_columns):

            address = (
                array.ctypes.data
                + (row_start + i) * array.strides[0]
                + (column_start + j) * array.strides[1]
            )

            addresses.append(address)

    # Count transactions separately for each warp.
    for start in range(0, len(addresses), simulator.warp_size):

        warp_addresses = addresses[
            start:start + simulator.warp_size
        ]

        simulator.record_global_accesses(warp_addresses)


def tiled_matmul(A, B, tile_size=16, simulator=None):
    """
    Matrix multiplication using tiled computation.

    Loads matrix tiles into shared memory and reuses them
    to calculate multiple output elements.

    Handles dimensions that are not multiples of tile_size.
    """

    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")

    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions")

    if not isinstance(tile_size, (int, np.integer)):
        raise ValueError("Tile size must be an integer")

    if tile_size <= 0:
        raise ValueError("Tile size must be positive")

    rows = A.shape[0]
    inner = A.shape[1]
    columns = B.shape[1]

    C = np.zeros((rows, columns), dtype=np.float32)

    # Calculate the memory needed for the two input tiles.
    A_tile_bytes = tile_size * tile_size * A.itemsize
    B_tile_bytes = tile_size * tile_size * B.itemsize

    shared_memory_required = A_tile_bytes + B_tile_bytes

    if simulator is not None:
        simulator.allocate_shared(shared_memory_required)

    # Traverse the output matrix in tiles.
    for row_start in range(0, rows, tile_size):

        for column_start in range(0, columns, tile_size):

            output_tile = np.zeros(
                (tile_size, tile_size),
                dtype=np.float32
            )

            # Traverse the shared matrix dimension.
            for inner_start in range(0, inner, tile_size):

                # Determine the actual dimensions of each input tile.
                valid_rows = min(tile_size, rows - row_start)
                valid_inner = min(tile_size, inner - inner_start)
                valid_columns = min(tile_size, columns - column_start)

                # Load the required sections of A and B.
                A_tile = A[
                    row_start:row_start + valid_rows,
                    inner_start:inner_start + valid_inner
                ]

                B_tile = B[
                    inner_start:inner_start + valid_inner,
                    column_start:column_start + valid_columns
                ]

                if simulator is not None:

                    # Count warp-level transactions for loading A.
                    record_tile_accesses(
                        A,
                        row_start,
                        inner_start,
                        valid_rows,
                        valid_inner,
                        simulator
                    )

                    # Count warp-level transactions for loading B.
                    record_tile_accesses(
                        B,
                        inner_start,
                        column_start,
                        valid_inner,
                        valid_columns,
                        simulator
                    )

                # Multiply the tiles and accumulate the results.
                output_tile[:valid_rows, :valid_columns] += (
                    A_tile @ B_tile
                )

            # Store the completed output tile.
            valid_rows = min(tile_size, rows - row_start)
            valid_columns = min(tile_size, columns - column_start)

            C[
                row_start:row_start + valid_rows,
                column_start:column_start + valid_columns
            ] = output_tile[:valid_rows, :valid_columns]

    return C