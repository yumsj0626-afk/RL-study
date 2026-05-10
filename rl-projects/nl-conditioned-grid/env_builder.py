from env import NavGridEnv


def _inside_grid(cell, grid_size):
    return 0 <= cell[0] < grid_size[0] and 0 <= cell[1] < grid_size[1]


def build_env(spec: dict) -> NavGridEnv:
    """
    Convert a JSON command spec into a NavGridEnv instance.

    Edge cases:
    - goal=None -> ValueError
    - goal inside obstacles -> ValueError
    - start == goal -> ValueError
    - any referenced cell outside grid -> ValueError
    """
    grid_size = tuple(spec["grid_size"])
    start = tuple(spec.get("start", [0, 0]))
    raw_goal = spec.get("goal")
    if raw_goal is None:
        raise ValueError("Goal is underspecified")
    goal = tuple(raw_goal)
    obstacles = [tuple(o) for o in spec.get("obstacles", [])]
    soft_avoid = [tuple(s) for s in spec.get("soft_avoid", [])]
    preference = spec.get("preference", "default")

    if start == goal:
        raise ValueError("Start equals goal")
    if not _inside_grid(start, grid_size):
        raise ValueError(f"Start {start} is outside grid {grid_size}")
    if not _inside_grid(goal, grid_size):
        raise ValueError(f"Goal {goal} is outside grid {grid_size}")
    for cell in obstacles:
        if not _inside_grid(cell, grid_size):
            raise ValueError(f"Obstacle {cell} is outside grid {grid_size}")
    for cell in soft_avoid:
        if not _inside_grid(cell, grid_size):
            raise ValueError(f"Soft avoid cell {cell} is outside grid {grid_size}")
    if goal in obstacles:
        raise ValueError(f"Goal {goal} is inside obstacles")
    if start in obstacles:
        raise ValueError(f"Start {start} is inside obstacles")

    return NavGridEnv(
        grid_size=grid_size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        soft_avoid=soft_avoid,
        preference=preference,
    )
