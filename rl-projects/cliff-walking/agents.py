import numpy as np


class TabularAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.5,
        gamma=1.0,
        epsilon=0.1,
        seed=42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def act(self, state) -> int:
        """epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.Q[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def update(self, *args, **kwargs):
        raise NotImplementedError


class SARSAAgent(TabularAgent):
    def update(self, s, a, r, ns, na, done):
        target = r if done else r + self.gamma * self.Q[ns, na]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class QLearningAgent(TabularAgent):
    def update(self, s, a, r, ns, done):
        target = r if done else r + self.gamma * np.max(self.Q[ns])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
