"""game.py
Simple Nim environment. State is normalized scalar.
"""
import numpy as np

class Nim:
    def __init__(self, starting_stones=15, max_take=3):
        self.starting_stones = int(starting_stones)
        self.max_take = int(max_take)
        self.reset()

    def reset(self):
        self.stones = self.starting_stones
        return self._get_state()

    def _get_state(self):
        return np.array([self.stones / max(1, self.starting_stones)], dtype=np.float32)

    def valid_actions(self):
        return [a for a in range(1, self.max_take + 1) if a <= self.stones]

    def step(self, action):
        action = int(max(1, min(action, self.stones)))
        self.stones -= action
        done = (self.stones == 0)
        return self._get_state(), 0.0, done
