"""오프라인 회귀 검증 (API 불필요, 비용 0).

test_cases/specs/*.json 의 모든 spec을 plan -> simulate -> render 까지 돌리고,
- forbidden 영역 충돌 여부를 검사하고
- spec별 interpretation_<name>.md 와 전체 summary, failure_taxonomy.md 를 생성한다.
나아가 핵심 불변식(assert)을 검사해 기하 스택이 깨지지 않았는지 잠근다.

그리드 프로젝트의 run_experiments.py + failure_taxonomy.md + interpretation 자동생성 패턴 계승.
"""
import json
import sys
from pathlib import Path

from controller import simulate
from nl_parser import validate_spec
from planner import PlanningError, _region_contains, plan_path
from preview_2d import RESULTS_DIR, render, render_failure

BASE_DIR = Path(__file__).resolve().parent
SPECS_DIR = BASE_DIR / "test_cases" / "specs"


def collisions(spec: dict, traj) -> int:
    """forbidden 영역(실제 기하, 팽창 0) 안에 들어간 궤적 점 수."""
    forbidden = spec.get("forbidden_regions", [])
    if not forbidden:
        return 0
    n = 0
    for (x, y, _yaw) in traj:
        if any(_region_contains(r, x, y, 0.0) for r in forbidden):
            n += 1
    return n


def path_length(traj) -> float:
    total = 0.0
    for a, b in zip(traj, traj[1:]):
        total += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return total


def run_one(spec_path: Path) -> dict:
    name = spec_path.stem
    spec = validate_spec(json.loads(spec_path.read_text(encoding="utf-8")))
    rec = {"name": name, "stage": None, "notes": spec.get("interpretation_notes", "")}

    try:
        waypoints = plan_path(spec)
    except PlanningError as e:
        rec.update(stage="planning_failed", success=False, reason=str(e),
                   waypoints=0, collisions=0, path_len=0.0)
        return rec, spec, None, None

    result = simulate(spec, waypoints)
    coll = collisions(spec, result["traj"])
    rec.update(
        stage="control",
        success=bool(result["success"]) and coll == 0,
        reason=result["reason"] if coll == 0 else f"{coll} collision steps",
        waypoints=len(waypoints),
        collisions=coll,
        path_len=round(path_length(result["traj"]), 2),
        speed=spec.get("speed"),
        preference=spec.get("preference"),
    )
    return rec, spec, waypoints, result


def write_interpretation(rec, spec):
    lines = [
        f"# {rec['name']}",
        "",
        f"- 해석 노트: {rec['notes']}",
        f"- 단계 결과: **{rec['stage']}**",
        f"- 성공: {rec['success']}  ({rec['reason']})",
        f"- 목표: {spec.get('goal')}  / 속도: {spec.get('speed')} / preference: {spec.get('preference')}",
        f"- forbidden: {len(spec.get('forbidden_regions', []))}개, "
        f"soft_avoid: {len(spec.get('soft_avoid_regions', []))}개",
        f"- waypoint 수: {rec['waypoints']}, 경로 길이: {rec['path_len']} m, "
        f"forbidden 충돌: {rec['collisions']}회",
    ]
    (RESULTS_DIR / f"interpretation_{rec['name']}.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec_files = sorted(SPECS_DIR.glob("*.json"))
    if not spec_files:
        print("no specs found in", SPECS_DIR)
        return

    records = []
    for sp in spec_files:
        rec, spec, wps, result = run_one(sp)
        records.append(rec)
        write_interpretation(rec, spec)
        if wps is not None:
            render(spec, wps, result, rec["name"], None)
        else:
            render_failure(spec, rec["name"], rec["reason"])
        status = "OK" if rec["success"] else ("FAIL-EXPECTED" if rec["stage"] == "planning_failed" else "FAIL")
        print(f"[{status:14}] {rec['name']:16} stage={rec['stage']:16} "
              f"coll={rec['collisions']} len={rec['path_len']} :: {rec['reason']}")

    (RESULTS_DIR / "all_experiments.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # failure taxonomy
    fails = [r for r in records if r["stage"] == "planning_failed"]
    tax = ["# Failure Taxonomy (offline)", ""]
    tax.append("계획 단계에서 의도적으로 중단된(controlled) 케이스:")
    tax += [f"- **{r['name']}**: {r['reason']}" for r in fails] or ["- (없음)"]
    (RESULTS_DIR / "failure_taxonomy.md").write_text("\n".join(tax), encoding="utf-8")

    # ---- 핵심 불변식 잠금 ----
    by = {r["name"]: r for r in records}
    failed_asserts = []
    if "center_block" in by:
        r = by["center_block"]
        if not (r["success"] and r["collisions"] == 0):
            failed_asserts.append("center_block 는 충돌 0 + 성공이어야 함")
    if "safe_detour" in by:
        r = by["safe_detour"]
        if not r["success"]:
            failed_asserts.append("safe_detour 는 성공이어야 함")
    if "underspecified" in by:
        r = by["underspecified"]
        if r["stage"] != "planning_failed":
            failed_asserts.append("underspecified(goal=null) 는 controlled planning_failed 여야 함")
    # 이름이 fail_* 인 spec 은 절대 깔끔히 성공하면 안 됨 (의도된 실패 케이스)
    for r in records:
        if r["name"].startswith("fail_") and r["success"]:
            failed_asserts.append(f"{r['name']} 는 실패해야 하는데 성공함")

    print()
    if failed_asserts:
        print("REGRESSION FAILED:")
        for a in failed_asserts:
            print("  -", a)
        sys.exit(1)
    print(f"ALL CHECKS PASSED ({len(records)} specs). results/ 에 그림·해석·요약 저장됨.")


if __name__ == "__main__":
    main()
