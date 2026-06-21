"""waypoint -> velocity command (go-to-goal) + 운동학 시뮬레이션.

이 컨트롤러가 만드는 (vx, vy, yaw_rate) 가 Phase 3 에서 Isaac Lab Go2 locomotion 정책의
velocity command 입력으로 들어간다. 여기서는 Isaac 없이 단순 unicycle 운동학으로 미리 검증한다.

body-frame 명령: vx=전진, vy=횡, yaw_rate=회전. preview 에서는 vy=0(전진+회전)으로 단순화.
"""
import math

# speed 키워드 -> 전진 속도 명령 크기 (m/s). Go2 flat 태스크의 command 범위(~[-1,1])를 넘지 않게.
SPEED_TO_VMAX = {"slow": 0.4, "normal": 0.7, "fast": 1.0}

# 컨트롤러 게인/한계
K_YAW = 2.0          # 헤딩 오차 -> yaw_rate 비례 게인
YAW_RATE_MAX = 1.0   # rad/s
REACH_RADIUS = 0.30  # 중간 waypoint 도달 판정 반경 (m)
GOAL_TOL = 0.25      # 최종 목표 도달 판정 (m)


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def go_to_goal(pose, target, vmax):
    """현재 pose=(x,y,yaw) 와 목표점 target=(x,y) -> (vx, vy, yaw_rate).

    헤딩이 어긋나면 제자리 회전 위주, 정렬되면 전진. 실제 로봇/정책에 줄 명령과 동일한 형태.
    """
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    desired_yaw = math.atan2(dy, dx)
    err = wrap_to_pi(desired_yaw - yaw)

    yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, K_YAW * err))
    # 헤딩이 정렬된 정도만큼만 전진(어긋나면 거의 제자리 회전)
    vx = vmax * max(0.0, math.cos(err))
    vy = 0.0
    return vx, vy, yaw_rate


def simulate(spec: dict, waypoints, dt: float = 0.05, max_time: float = 80.0):
    """waypoint 추종을 unicycle 운동학으로 시뮬레이션.

    Returns dict: traj[(x,y,yaw)], cmds[(vx,vy,yaw_rate)], times, success, reason, wp_index.
    """
    vmax = SPEED_TO_VMAX.get(spec.get("speed", "normal"), 0.7)
    start = spec.get("start", [0.5, 0.5])
    goal = waypoints[-1]

    # 초기 yaw: 첫 목표 방향을 바라보게
    first = waypoints[1] if len(waypoints) > 1 else goal
    yaw0 = math.atan2(first[1] - start[1], first[0] - start[0])
    x, y, yaw = start[0], start[1], yaw0

    traj = [(x, y, yaw)]
    cmds = []
    times = [0.0]
    wp_idx = 1  # 0 은 시작점
    t = 0.0
    success = False
    reason = "max_time reached without reaching goal"

    steps = int(max_time / dt)
    for _ in range(steps):
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        vx, vy, yaw_rate = go_to_goal((x, y, yaw), target, vmax)

        # body-frame -> world 적분
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw = wrap_to_pi(yaw + yaw_rate * dt)
        t += dt

        cmds.append((vx, vy, yaw_rate))
        traj.append((x, y, yaw))
        times.append(t)

        # waypoint 전진 / 목표 도달 판정
        if wp_idx < len(waypoints) - 1:
            if math.hypot(target[0] - x, target[1] - y) < REACH_RADIUS:
                wp_idx += 1
        else:
            if math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOL:
                success = True
                reason = "goal reached"
                break

    return {
        "traj": traj,
        "cmds": cmds,
        "times": times,
        "success": success,
        "reason": reason,
        "vmax": vmax,
        "wp_index": wp_idx,
    }


if __name__ == "__main__":
    from planner import plan_path

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
    result = simulate(demo, wps)
    print(f"success={result['success']} reason={result['reason']} "
          f"steps={len(result['cmds'])} final={result['traj'][-1]}")
