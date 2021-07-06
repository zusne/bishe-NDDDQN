import numpy as np
import random

class ReplayMemory:
    """Store and replay (sample) memories."""
    def __init__(self,
                max_size,
                window_size,
                input_shape):
        """Specify the maximum size of the memory. Once the memory fills up oldest values are removed."""
        self._max_size = max_size
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self._memory = []


    def append(self, old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal):
        """Add a list of samples to the replay memory."""
        if len(self._memory) >= self._max_size:
            del(self._memory[0])

        self._memory.append((old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal))


    def sample(self, batch_size):
        """Return samples from the memory.
        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list, frequency_list)
        """
        samples = random.sample(self._memory, min(batch_size, len(self._memory)))
        zipped = list(zip(*samples))
        zipped[0] = np.reshape(zipped[0], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        zipped[9] = np.reshape(zipped[9], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        return zipped
