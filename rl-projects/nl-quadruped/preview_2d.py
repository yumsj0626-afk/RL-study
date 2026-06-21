"""Phase 0 통합 검증: 자연어/spec -> 경로 -> 속도명령 전체를 Isaac 없이 그림으로 확인.

사용법:
    # 오프라인(LLM 없음, API 비용 0) — 예제 spec 사용
    python preview_2d.py --spec test_cases/specs/center_block.json

    # 자연어 명령(OpenAI 호출, .env 의 OPENAI_API_KEY 필요)
    python preview_2d.py "오른쪽 위 구석으로 가되 중앙은 절대 피해서 천천히"

결과: results/preview_<name>.png (좌: 월드+경로+실행궤적, 우: 속도명령 시계열) + spec/요약 저장.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from controller import simulate
from nl_parser import validate_spec
from planner import PlanningError, plan_path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def _draw_region(ax, region, facecolor, edgecolor, label=None):
    if region["type"] == "box":
        mnx, mny = region["min"]
        mxx, mxy = region["max"]
        ax.add_patch(patches.Rectangle((mnx, mny), mxx - mnx, mxy - mny,
                                       facecolor=facecolor, edgecolor=edgecolor,
                                       alpha=0.45, label=label))
    elif region["type"] == "circle":
        cx, cy = region["center"]
        ax.add_patch(patches.Circle((cx, cy), region["radius"],
                                    facecolor=facecolor, edgecolor=edgecolor,
                                    alpha=0.45, label=label))


def _draw_world(ax, spec: dict):
    """월드 경계 + forbidden/soft 영역 + start/goal 마커 (궤적 제외)."""
    wx, wy = spec["world_size"]
    ax.set_xlim(0, wx)
    ax.set_ylim(0, wy)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.grid(True, alpha=0.3)
    for i, r in enumerate(spec.get("forbidden_regions", [])):
        _draw_region(ax, r, "red", "darkred", "forbidden" if i == 0 else None)
    for i, r in enumerate(spec.get("soft_avoid_regions", [])):
        _draw_region(ax, r, "orange", "darkorange", "soft avoid" if i == 0 else None)
    start = spec.get("start", [0.5, 0.5])
    ax.plot(start[0], start[1], "s", color="green", markersize=11, label="start")
    goal = spec.get("goal")
    if goal is not None:
        ax.plot(goal[0], goal[1], "*", color="gold", markersize=20,
                markeredgecolor="k", label="goal")


def render_failure(spec: dict, name: str, reason: str):
    """계획 실패 케이스의 증거 이미지(궤적 없음). 실패도 보고서 증거로 남긴다."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_title("World (planning failed)")
    _draw_world(ax, spec)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"[PLANNING FAILED] {name}\n{reason}", fontsize=11, color="darkred")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = RESULTS_DIR / f"preview_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render(spec: dict, waypoints, result, name: str, command: str | None):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ---- 좌: 월드 + 경로 + 실행 궤적 ----
    ax.set_title("World / Path / Executed trajectory")
    _draw_world(ax, spec)
    ax.plot([w[0] for w in waypoints], [w[1] for w in waypoints],
            "o--", color="tab:blue", alpha=0.7, label="planned waypoints")
    ax.plot([p[0] for p in result["traj"]], [p[1] for p in result["traj"]],
            "-", color="tab:green", linewidth=2, label="executed trajectory")
    ax.legend(loc="best", fontsize=8)

    # ---- 우: 속도명령 시계열 ----
    times = result["times"][1:]  # cmds 와 길이 맞춤
    cmds = result["cmds"]
    ax2.plot(times, [c[0] for c in cmds], label="vx (forward) [m/s]")
    ax2.plot(times, [c[2] for c in cmds], label="yaw_rate [rad/s]")
    ax2.set_title("Velocity command to locomotion policy")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("command")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    status = "SUCCESS" if result["success"] else "FAIL"
    sub = f"[{status}] speed={spec.get('speed')} pref={spec.get('preference')} " \
          f"vmax={result['vmax']} | {result['reason']}"
    if command:
        sub = f'"{command}"\n' + sub
    fig.suptitle(sub, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = RESULTS_DIR / f"preview_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", help="자연어 명령 (생략하고 --spec 사용 가능)")
    ap.add_argument("--spec", help="JSON spec 파일 경로 (LLM 호출 생략, 오프라인 테스트)")
    ap.add_argument("--name", help="결과 파일 이름 태그")
    args = ap.parse_args()

    command = args.command
    if args.spec:
        spec = validate_spec(json.loads(Path(args.spec).read_text(encoding="utf-8")))
        name = args.name or Path(args.spec).stem
    elif command:
        from nl_parser import parse_command
        spec = parse_command(command)
        name = args.name or "cmd"
        print(json.dumps(spec, ensure_ascii=False, indent=2))
    else:
        ap.error("자연어 명령 또는 --spec 중 하나는 필요합니다.")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"spec_{name}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    # 계획 단계 실패는 controlled failure 로 보고(그리드 프로젝트의 실패 기록 계승)
    try:
        waypoints = plan_path(spec)
    except PlanningError as e:
        print(f"[PLANNING FAILED] {e}")
        print("  -> 이는 의도된 실패 처리입니다(goal=null 등). Phase 3에서 재질문 루프로 연결.")
        return

    result = simulate(spec, waypoints)
    out = render(spec, waypoints, result, name, command)
    print(f"success={result['success']} reason={result['reason']} "
          f"waypoints={len(waypoints)} steps={len(result['cmds'])}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
