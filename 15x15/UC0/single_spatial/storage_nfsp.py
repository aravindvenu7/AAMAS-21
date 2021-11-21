from collections import deque
import random, math, itertools
import numpy as np

class ReplayMemory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)

class SLMemory(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action):
        """Add an experience to the buffer."""
        self.buffer.append((state,action))
    
    def sample(self, batch_size):
        # Efficient Reservoir Sampling
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        buffer_size = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while (idx < buffer_size and idx <= threshold):
            m = random.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1
        
        while (idx < buffer_size):
            p = float(batch_size) / idx
            u = random.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < buffer_size:
                k = random.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1
        state, action = zip(*random.sample(self.buffer, batch_size))
        
        return state, action
    
    def __len__(self):
        return len(self.buffer)        