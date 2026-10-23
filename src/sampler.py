from typing import Sized

import numpy as np
import torch
from torch.utils.data import Sampler


class ShuffleBufferSampler(Sampler[int]):
    """
    Approximates a full per-epoch shuffle for datasets too large to permute in memory.

    DataLoader's default shuffle=True uses RandomSampler, which builds a torch.randperm(n)
    and converts it to a Python list -- for a large corpus (such as the Huggingface Wikipedia
    dataset, which creates ~1.9 billion token positions), that's tens of GB just for one 
    epoch's index order, independent of num_workers or batch_size. This sampler instead keeps 
    a fixed-size window of "live" indices: each draw picks a random slot in that window, yields
    it, and refills the slot with the next not-yet-seen index. Memory is O(buffer_size), not 
    O(len(dataset)) -- the same tradeoff tf.data.Dataset.shuffle makes for the same reason. 
    Randomization is local to a buffer_size-wide window rather than a true global permutation,
    which is fine for SGD training but means two indices more than buffer_size apart in the 
    original order can never land in the same batch.
    """

    def __init__(self, data_source: Sized, buffer_size: int = 1_000_000, seed: int = None):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.seed = seed

    def __len__(self):
        return len(self.data_source)

    # Time: O(n), Space: O(buffer_size)   (n = len(data_source); space independent of n -- that's the point)
    def __iter__(self):
        n = len(self.data_source)
        rng = np.random.default_rng(self.seed)
        buffer_len = min(self.buffer_size, n)
        buffer = np.arange(buffer_len, dtype=np.int64)
        next_idx = buffer_len

        # Steady state: the buffer stays full. Draw random slot indices in chunks (not all
        # n at once -- that would just recreate the same O(n) memory problem one level down).
        chunk_size = min(buffer_len, 1_000_000)
        steady_draws = n - buffer_len
        drawn = 0
        while drawn < steady_draws:
            chunk = min(chunk_size, steady_draws - drawn)
            for slot in rng.integers(0, buffer_len, size=chunk, dtype=np.int64):
                yield int(buffer[slot])
                buffer[slot] = next_idx
                next_idx += 1
            drawn += chunk

        # Drain: no more unseen indices to refill with. Finish with a Fisher-Yates shuffle
        # of what's left in the buffer, so the last buffer_len draws are still exactly
        # uniformly random instead of just whatever order happened to be left over.
        for live_len in range(buffer_len, 0, -1):
            slot = int(rng.integers(0, live_len))
            yield int(buffer[slot])
            buffer[slot] = buffer[live_len - 1]


class NegativeSampler:
    """
    Creates a sample of negative context words for use when training.
    """
    # Time: O(V), Space: O(V)   (V = vocab_size; one-time setup, not called per batch)
    def __init__(self, word_counts: list[int], power: float = 0.75, device: torch.device = None):
        # Calculate the probability sampling for the whole vocabulary.
        # This method is similar to the process in word2vec, where we decrease
        # the importance of word frequency for use as a negative context word
        # by raising it to the power of 0.75 (default).

        # First, set the device appropriately if not provided. Cuda is prioritized over CPU.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Calculate probability weights
        counts = torch.tensor(word_counts, dtype=torch.float32)
        adjusted_counts = counts ** power
        self.weights = adjusted_counts / torch.sum(adjusted_counts)

        # Move weights to the appropriate device
        self.weights = self.weights.to(self.device)

        # Cached cumulative sum for use in sample()
        self.cumsum = torch.cumsum(self.weights, dim=0)

    # Time: O(num_samples * log V), Space: O(num_samples)   (V = vocab_size, via binary search into the cached cumsum)
    def sample(self, num_samples: int) -> torch.Tensor:
        # torch.searchsorted does binary search over the cached cumsum, using u as a random sample selection
        u = torch.rand(num_samples, device=self.device) * self.cumsum[-1]
        return torch.searchsorted(self.cumsum, u)