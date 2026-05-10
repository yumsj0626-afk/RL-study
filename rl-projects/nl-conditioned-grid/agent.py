import numpy as np


class QLearningAgent:
    def __init__(
        self,
        grid_size,
        n_actions=4,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    ):
        self.grid_size = tuple(grid_size)
        self.n_states = self.grid_size[0] * self.grid_size[1]
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.n_states, n_actions), dtype=float)

    def _state_to_idx(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def act(self, state, greedy=False):
        idx = self._state_to_idx(state)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.Q[idx]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def update(self, s, a, r, ns, done):
        s_idx = self._state_to_idx(s)
        ns_idx = self._state_to_idx(ns)
        target = r if done else r + self.gamma * np.max(self.Q[ns_idx])
        self.Q[s_idx, a] += self.alpha * (target - self.Q[s_idx, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
