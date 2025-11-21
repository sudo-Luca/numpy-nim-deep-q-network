import random

class Game:
    def __init__(self, env=None, agent=None, heaps=None, max_heap=None, opponent_policy=None):
        self.env = env
        self.agent = agent
        self.heaps = list(heaps) if heaps is not None else [3, 4, 5]
        self.max_heap = max_heap or max(self.heaps)
        # opponent_policy receives current heaps and must return an action (heap_idx, remove)
        self.opponent_policy = opponent_policy or (lambda heaps: random.choice(self.legal_actions(heaps)))
        self.current_player = 0

    def reset(self, heaps=None):
        self.heaps = list(heaps) if heaps is not None else [3, 4, 5]
        self.max_heap = max(self.heaps)
        self.current_player = 0
        return self.get_state()

    def get_state(self):
        # state is immutable tuple of heap sizes and current player
        return (tuple(self.heaps), self.current_player)

    def legal_actions(self, heaps=None):
        heaps = heaps if heaps is not None else self.heaps
        actions = []
        for i, h in enumerate(heaps):
            for remove in range(1, h + 1):
                actions.append((i, remove))
        return actions

    def _apply_action(self, action):
        # accepts either (heap_idx, remove) or an encoded int:
        # encoded int = heap_idx * max_heap + (remove-1)
        if isinstance(action, int):
            heap_idx = action // self.max_heap
            remove = (action % self.max_heap) + 1
            action = (heap_idx, remove)
        heap_idx, remove = action
        if heap_idx < 0 or heap_idx >= len(self.heaps) or remove <= 0 or remove > self.heaps[heap_idx]:
            raise ValueError("Illegal action")
        self.heaps[heap_idx] -= remove

    def step(self, action):
        # Agent (player 0) move
        try:
            self._apply_action(action)
        except ValueError:
            # illegal action: strong negative reward and terminate
            return self.get_state(), -1.0, True, {"illegal_action": True}
        # check terminal after agent move
        if all(h == 0 for h in self.heaps):
            return self.get_state(), 1.0, True, {"winner": 0}
        # Opponent move (player 1) using policy provided at init (defaults to random)
        opp_action = self.opponent_policy(list(self.heaps))
        try:
            self._apply_action(opp_action)
        except ValueError:
            # opponent made illegal move: treat as agent win
            return self.get_state(), 1.0, True, {"opponent_illegal": True}
        # check terminal after opponent move
        if all(h == 0 for h in self.heaps):
            return self.get_state(), -1.0, True, {"winner": 1}
        # game continues
        return self.get_state(), 0.0, False, {}

    def action_space_size(self):
        # discrete action encoding size: num_heaps * max_heap
        return len(self.heaps) * self.max_heap

    def render(self):
        return "Heaps: " + ", ".join(str(h) for h in self.heaps)