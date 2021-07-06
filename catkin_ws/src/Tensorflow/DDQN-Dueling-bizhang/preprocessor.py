import numpy as np

class HistoryPreprocessor:
    """
    Keeps the last k states.
    history_length (int): Number of states to keep.
    """

    def __init__(self, input_shape, history_length=1):
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self.history = np.zeros(shape=(self._WIDTH, self._HEIGHT, history_length))
        self.history_length = history_length

    def process_state_for_memory(self, state):
        self.history[:,:,1:]=self.history[:,:,:self.history_length-1]
        self.history[:,:,0]=state

    def reset(self):
        self.history = np.zeros(shape=(self._WIDTH, self._HEIGHT, self.history_length))

    def get_state(self):
        return self.history.copy()