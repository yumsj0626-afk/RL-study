"""spec -> occupancy grid -> A* -> waypoint 리스트.

연속 월드(미터)를 해상도 `res` 의 점유 격자로 이산화하고, 8방향 A* 로 경로를 찾은 뒤
line-of-sight 단순화로 waypoint 수를 줄인다. nl-conditioned-grid 의 격자 계획 로직을 계승한 부분.

좌표계: [x, y], x=east, y=north, 원점 [0,0] = bottom-left.
"""
import heapq
import math

import numpy as np


class PlanningError(ValueError):
    """경로 계획 실패 (목표 미명시 / 시작·목표가 막힘 / 경로 없음)."""


def _region_contains(region: dict, x: float, y: float, inflate: float) -> bool:
    if region["type"] == "box":
        mnx, mny = region["min"]
        mxx, mxy = region["max"]
        return (mnx - inflate) <= x <= (mxx + inflate) and (mny - inflate) <= y <= (mxy + inflate)
    if region["type"] == "circle":
        cx, cy = region["center"]
        return math.hypot(x - cx, y - cy) <= region["radius"] + inflate
    raise ValueError(f"Unknown region type: {region['type']}")


def build_occupancy(spec: dict, res: float, robot_radius: float):
    """forbidden_regions 를 (clearance + robot_radius) 만큼 팽창시켜 막힌 격자 생성.

    Returns: (blocked[nx, ny] bool, nx, ny)  — blocked[ix, iy] == True 면 통행 불가.
    """
    wx, wy = spec["world_size"]
    nx = int(math.ceil(wx / res))
    ny = int(math.ceil(wy / res))
    inflate = spec.get("clearance", 0.0) + robot_radius

    blocked = np.zeros((nx, ny), dtype=bool)
    forbidden = spec.get("forbidden_regions", [])
    if forbidden:
        for ix in range(nx):
            cx = (ix + 0.5) * res
            for iy in range(ny):
                cy = (iy + 0.5) * res
                if any(_region_contains(r, cx, cy, inflate) for r in forbidden):
                    blocked[ix, iy] = True
    return blocked, nx, ny


def _soft_mask(spec: dict, res: float, nx: int, ny: int) -> np.ndarray:
    """soft_avoid 영역에 속한 셀 마스크(preference 무관)."""
    mask = np.zeros((nx, ny), dtype=bool)
    soft = spec.get("soft_avoid_regions", [])
    if not soft:
        return mask
    for ix in range(nx):
        cx = (ix + 0.5) * res
        for iy in range(ny):
            cy = (iy + 0.5) * res
            if any(_region_contains(r, cx, cy, 0.0) for r in soft):
                mask[ix, iy] = True
    return mask


def _world_to_cell(x, y, res, nx, ny):
    ix = min(max(int(x / res), 0), nx - 1)
    iy = min(max(int(y / res), 0), ny - 1)
    return ix, iy


def _cell_to_world(ix, iy, res):
    return ((ix + 0.5) * res, (iy + 0.5) * res)


def _line_of_sight(blocked, a, b, res, step):
    """월드 좌표 a->b 직선이 막힌 격자를 통과하지 않는지 샘플링 검사."""
    (ax, ay), (bx, by) = a, b
    dist = math.hypot(bx - ax, by - ay)
    n = max(1, int(dist / step))
    nx, ny = blocked.shape
    for i in range(n + 1):
        t = i / n
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t
        ix = min(max(int(x / res), 0), nx - 1)
        iy = min(max(int(y / res), 0), ny - 1)
        if blocked[ix, iy]:
            return False
    return True


def plan_path(spec: dict, res: float = 0.1, robot_radius: float = 0.25):
    """spec -> waypoint 리스트 [[x,y], ...] (start 포함, goal 포함).

    Raises PlanningError 로 단계별 실패를 노출(그리드 프로젝트의 build/learning 실패 기록 계승).
    """
    if spec.get("goal") is None:
        raise PlanningError("Goal is underspecified (goal=null)")

    blocked, nx, ny = build_occupancy(spec, res, robot_radius)
    is_safe = spec.get("preference") == "safe"
    soft_mask = _soft_mask(spec, res, nx, ny)
    # preference=safe 일 때만 soft 영역에 A* 추가비용을 매기고, 단순화에서도 soft를 회피 대상에 포함
    soft_cost = soft_mask.astype(float) * 5.0 if is_safe else np.zeros_like(soft_mask, dtype=float)
    los_block = (blocked | soft_mask) if is_safe else blocked

    start = tuple(spec.get("start", [0.5, 0.5]))
    goal = tuple(spec["goal"])
    s = _world_to_cell(start[0], start[1], res, nx, ny)
    g = _world_to_cell(goal[0], goal[1], res, nx, ny)

    if blocked[s]:
        raise PlanningError(f"Start {start} is inside an inflated forbidden region")
    if blocked[g]:
        raise PlanningError(f"Goal {goal} is inside an inflated forbidden region")
    if s == g:
        raise PlanningError("Start equals goal")

    # 8방향 A*
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(c):
        return math.hypot(c[0] - g[0], c[1] - g[1])

    open_heap = [(h(s), 0.0, s)]
    came_from = {}
    g_score = {s: 0.0}
    closed = set()

    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == g:
            break
        if cur in closed:
            continue
        closed.add(cur)
        for dx, dy in neighbors:
            ncx, ncy = cur[0] + dx, cur[1] + dy
            if not (0 <= ncx < nx and 0 <= ncy < ny):
                continue
            if blocked[ncx, ncy]:
                continue
            # 대각 이동 시 모서리 관통 방지
            if dx != 0 and dy != 0 and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue
            step_cost = math.hypot(dx, dy) + soft_cost[ncx, ncy]
            tentative = gc + step_cost
            nxt = (ncx, ncy)
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came_from[nxt] = cur
                heapq.heappush(open_heap, (tentative + h(nxt), tentative, nxt))

    if g not in came_from and g != s:
        raise PlanningError("No collision-free path from start to goal")

    # 경로 복원 (cell 시퀀스)
    path_cells = [g]
    cur = g
    while cur != s:
        cur = came_from[cur]
        path_cells.append(cur)
    path_cells.reverse()

    # cell -> world, 실제 start/goal 좌표를 양 끝에 사용
    pts = [list(start)]
    pts += [list(_cell_to_world(ix, iy, res)) for (ix, iy) in path_cells[1:-1]]
    pts.append(list(goal))

    # line-of-sight 단순화 (string pulling): waypoint 수 축소
    simplified = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_of_sight(los_block, pts[i], pts[j], res, res / 2):
            j -= 1
        simplified.append(pts[j])
        i = j
    return simplified


if __name__ == "__main__":
    demo = {
        "world_size": [8.0, 8.0],
        "start": [0.5, 0.5],
        "goal": [7.5, 7.5],
        "forbidden_regions": [{"type": "box", "min": [3.0, 3.0], "max": [5.0, 5.0]}],
        "soft_avoid_regions": [],
        "clearance": 0.3,
        "speed": "slow",
        "preference": "default",
    }
    wps = plan_path(demo)
    print(f"{len(wps)} waypoints:")
    for w in wps:
        print(f"  ({w[0]:.2f}, {w[1]:.2f})")
