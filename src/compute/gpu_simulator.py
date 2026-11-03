class GPUSimulator:

    def __init__(
        self,
        shared_memory_limit=48 * 1024,
        warp_size=32,
        transaction_size=32,
        max_threads_per_block=1024
    ):
        if shared_memory_limit <= 0:
            raise ValueError("Shared memory limit must be positive")

        if warp_size <= 0 or transaction_size <= 0:
            raise ValueError("Warp and transaction sizes must be positive")

        if max_threads_per_block <= 0:
            raise ValueError("Maximum threads must be positive")

        if max_threads_per_block < warp_size:
            raise ValueError("Maximum threads must be at least one warp")

        self.shared_memory_limit = shared_memory_limit
        self.warp_size = warp_size
        self.transaction_size = transaction_size
        self.max_threads_per_block = max_threads_per_block

        self.global_memory_transactions = 0
        self.shared_memory_used = 0

    def record_global_accesses(self, byte_addresses):
        """
        Count transactions for one group's memory accesses.

        Addresses in the same transaction-sized segment
        require only one transaction.
        """

        if len(byte_addresses) == 0:
            return 0

        segments = set()

        for address in byte_addresses:

            if not isinstance(address, (int, float)):
                raise ValueError("Memory addresses must be numeric")

            if address < 0 or int(address) != address:
                raise ValueError("Memory addresses must be non-negative integers")

            segment = int(address) // self.transaction_size
            segments.add(segment)

        transactions = len(segments)

        self.global_memory_transactions += transactions

        return transactions

    def allocate_shared(self, bytes_requested):
        """
        Allocate shared memory for a simulated block.
        """

        if bytes_requested < 0:
            raise ValueError("Allocation size cannot be negative")

        if bytes_requested > self.shared_memory_limit:
            raise MemoryError("Shared memory limit exceeded")

        self.shared_memory_used = bytes_requested

        return True

    def thread_index(self, block_index, thread_index, block_size=None):
        """
        Calculate a thread's global index.

        block_size defaults to the maximum supported block size.
        """

        if block_size is None:
            block_size = self.max_threads_per_block

        if block_index < 0 or thread_index < 0:
            raise ValueError("Thread indices cannot be negative")

        if block_size <= 0 or block_size > self.max_threads_per_block:
            raise ValueError("Invalid block size")

        if thread_index >= block_size:
            raise ValueError("Thread index exceeds block size")

        return block_index * block_size + thread_index

    def is_valid_thread(self, global_index, total_elements):
        """
        Check whether a thread corresponds to a valid element.
        """

        return 0 <= global_index < total_elements

    def get_stats(self):
        """
        Return current simulator statistics.
        """

        return {
            "global_memory_transactions":
                self.global_memory_transactions,

            "shared_memory_used":
                self.shared_memory_used,

            "shared_memory_limit":
                self.shared_memory_limit
        }

    def reset(self):
        """
        Reset accumulated statistics.
        """

        self.global_memory_transactions = 0
        self.shared_memory_used = 0