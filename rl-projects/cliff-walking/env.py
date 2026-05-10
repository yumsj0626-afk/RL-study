import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class CliffWalkingEnv:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }
    ACTION_MARKERS = {
        0: "^",
        1: ">",
        2: "v",
        3: "<",
    }

    def __init__(
        self,
        height=4,
        width=12,
        start=(3, 0),
        goal=(3, 11),
        cliff_cells=None,
    ):
        self.height = height
        self.width = width
        self.start = tuple(start)
        self.goal = tuple(goal)
        if cliff_cells is None:
            cliff_cells = [(height - 1, c) for c in range(1, width - 1)]
        self.cliff_cells = {tuple(cell) for cell in cliff_cells}
        self.n_actions = 4
        self.state = self.start

    def reset(self) -> tuple:
        """returns initial state (row, col)"""
        self.state = self.start
        return self.state

    def step(self, action: int) -> tuple:
        """returns (next_state, reward, done, info)"""
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        dr, dc = self.ACTIONS[action]
        row = min(max(self.state[0] + dr, 0), self.height - 1)
        col = min(max(self.state[1] + dc, 0), self.width - 1)
        candidate = (row, col)

        if candidate in self.cliff_cells:
            self.state = self.start
            return self.state, -100, False, {"event": "cliff"}

        self.state = candidate
        if self.state == self.goal:
            return self.state, 0, True, {"event": "goal"}
        return self.state, -1, False, {"event": "step"}

    def render_policy(self, Q, save_path=None):
        """matplotlib visualization of the greedy policy."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linewidth=1)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))

        for row, col in self.cliff_cells:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="#111111"))
            ax.text(col, row, "C", color="white", ha="center", va="center", fontsize=10)

        sr, sc = self.start
        gr, gc = self.goal
        ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="#7fc97f", alpha=0.7))
        ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color="#fdc086", alpha=0.8))
        ax.text(sc, sr, "S", ha="center", va="center", fontweight="bold")
        ax.text(gc, gr, "G", ha="center", va="center", fontweight="bold")

        for row in range(self.height):
            for col in range(self.width):
                state = (row, col)
                if state in self.cliff_cells or state in (self.start, self.goal):
                    continue
                idx = row * self.width + col
                action = int(np.argmax(Q[idx]))
                ax.text(
                    col,
                    row,
                    self.ACTION_MARKERS[action],
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#2b5b84",
                )

        ax.set_title("Greedy Policy")
        fig.tight_layout()
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=160)
        plt.close(fig)
        return fig
