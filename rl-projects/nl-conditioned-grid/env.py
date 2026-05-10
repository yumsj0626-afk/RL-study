import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class NavGridEnv:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }
    ACTION_MARKERS = {0: "^", 1: ">", 2: "v", 3: "<"}

    def __init__(
        self,
        grid_size: tuple,
        start: tuple,
        goal: tuple,
        obstacles: list = None,
        soft_avoid: list = None,
        preference: str = "default",
        max_steps: int = 200,
    ):
        self.grid_size = tuple(grid_size)
        self.height, self.width = self.grid_size
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = {tuple(cell) for cell in (obstacles or [])}
        self.soft_avoid = {tuple(cell) for cell in (soft_avoid or [])}
        self.preference = preference
        self.max_steps = max_steps
        self.n_actions = 4
        self.state = self.start
        self.step_count = 0

    def reset(self) -> tuple:
        self.state = self.start
        self.step_count = 0
        return self.state

    def _clip(self, row, col):
        return (
            min(max(row, 0), self.height - 1),
            min(max(col, 0), self.width - 1),
        )

    def _manhattan_to_goal(self, state):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def _near_soft_avoid(self, state):
        return any(abs(state[0] - r) + abs(state[1] - c) <= 1 for r, c in self.soft_avoid)

    def step(self, action: int) -> tuple:
        """
        Reward rules:
        - goal: +100, done=True
        - obstacle: -100 and reset to start, done=False unless timeout
        - soft_avoid: -5
        - preference="shortest": add distance-delta shaping * 0.5
        - preference="safe": add -2 near soft_avoid cells
        - otherwise: -1 time penalty
        - max_steps: done=True with timeout flag
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        old_state = self.state
        dr, dc = self.ACTIONS[action]
        candidate = self._clip(old_state[0] + dr, old_state[1] + dc)
        self.step_count += 1

        info = {"event": "step", "timeout": False}
        done = False

        if candidate in self.obstacles:
            self.state = self.start
            reward = -100
            info["event"] = "obstacle"
        else:
            self.state = candidate
            if self.state == self.goal:
                reward = 100
                done = True
                info["event"] = "goal"
            elif self.state in self.soft_avoid:
                reward = -5
                info["event"] = "soft_avoid"
            else:
                reward = -1

            if not done and self.preference == "shortest":
                old_dist = self._manhattan_to_goal(old_state)
                new_dist = self._manhattan_to_goal(self.state)
                reward += (old_dist - new_dist) * 0.5
            if not done and self.preference == "safe" and self._near_soft_avoid(self.state):
                reward -= 2

        if self.step_count >= self.max_steps and not done:
            done = True
            info["timeout"] = True
            info["event"] = "timeout"

        return self.state, reward, done, info

    def render(self, Q=None, save_path=None, title=""):
        """Visualize grid, constraints, goal, and optional greedy policy."""
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linewidth=1)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))

        for row, col in self.obstacles:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="#111111"))
            ax.text(col, row, "X", color="white", ha="center", va="center", fontsize=12)
        for row, col in self.soft_avoid:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="#f3c969", alpha=0.9))
            ax.text(col, row, "!", color="#4a3b00", ha="center", va="center", fontsize=12)

        sr, sc = self.start
        gr, gc = self.goal
        ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="#7fc97f", alpha=0.75))
        ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color="#fdc086", alpha=0.9))
        ax.text(sc, sr, "S", ha="center", va="center", fontweight="bold")
        ax.text(gc, gr, "*", ha="center", va="center", fontweight="bold", fontsize=16)

        if Q is not None:
            for row in range(self.height):
                for col in range(self.width):
                    state = (row, col)
                    if state in self.obstacles or state == self.goal:
                        continue
                    idx = row * self.width + col
                    action = int(np.argmax(Q[idx]))
                    ax.text(
                        col,
                        row,
                        self.ACTION_MARKERS[action],
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#24527a",
                    )

        ax.set_title(title or "NavGridEnv")
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=160)
        plt.close(fig)
        return fig
