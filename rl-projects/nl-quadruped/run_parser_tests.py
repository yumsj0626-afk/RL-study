"""배치 파서 실험 (OpenAI 호출 — .env 의 OPENAI_API_KEY 필요).

test_cases/commands.json 의 모든 자연어 명령을 한 번에 파싱하고:
- spec_<id>.json 저장, 스키마 검증(parse_command 내부),
- goal/speed/preference/forbidden/soft 요약,
- 에러는 명령별로 잡아 배치를 계속 진행,
- results/parser_test_results.json + parser_test_analysis.md 생성.

옵션:
    python run_parser_tests.py              # 파싱만
    python run_parser_tests.py --preview    # 파싱 후 plan+sim+render 까지 end-to-end
    python run_parser_tests.py --only T02 T03   # 특정 id만

파서는 비결정적일 수 있으니 여러 번 돌려보며 interpretation_notes 를 점검하는 용도.
"""
import argparse
import json
from pathlib import Path

from nl_parser import parse_command

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CASES = BASE_DIR / "test_cases" / "commands.json"


def summarize(spec: dict) -> dict:
    return {
        "goal": spec.get("goal"),
        "speed": spec.get("speed"),
        "preference": spec.get("preference"),
        "n_forbidden": len(spec.get("forbidden_regions", [])),
        "n_soft": len(spec.get("soft_avoid_regions", [])),
        "notes": spec.get("interpretation_notes", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="파싱 후 plan+sim+render 까지 실행")
    ap.add_argument("--only", nargs="*", help="특정 id만 실행 (예: --only T02 T03)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    records = []
    for c in cases:
        cid, command, expected = c["id"], c["command"], c.get("expected", "")
        rec = {"id": cid, "command": command, "expected": expected}
        try:
            spec = parse_command(command)
            (RESULTS_DIR / f"spec_{cid}.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            rec["parse"] = "ok"
            rec.update(summarize(spec))

            if args.preview:
                from planner import PlanningError, plan_path
                from controller import simulate
                from preview_2d import render
                try:
                    wps = plan_path(spec)
                    result = simulate(spec, wps)
                    render(spec, wps, result, cid, command)
                    rec["e2e"] = f"success={result['success']} ({result['reason']})"
                except PlanningError as e:
                    rec["e2e"] = f"planning_failed: {e}"
        except Exception as e:  # 파싱/네트워크/검증 실패 — 배치 계속
            rec["parse"] = "ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"

        records.append(rec)
        tag = rec.get("parse")
        extra = rec.get("e2e", rec.get("error", ""))
        print(f"[{tag:5}] {cid} {command[:34]:34} goal={rec.get('goal')} "
              f"speed={rec.get('speed')} pref={rec.get('preference')} {extra}")

    (RESULTS_DIR / "parser_test_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # 분석표(markdown)
    rows = ["# Parser Test Analysis", "",
            "| id | command | expected | parse | goal | speed | pref | forb | soft | notes/error |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        detail = r.get("notes", "") if r.get("parse") == "ok" else r.get("error", "")
        rows.append(
            f"| {r['id']} | {r['command']} | {r['expected']} | {r['parse']} | "
            f"{r.get('goal')} | {r.get('speed','')} | {r.get('preference','')} | "
            f"{r.get('n_forbidden','')} | {r.get('n_soft','')} | {detail} |")
    (RESULTS_DIR / "parser_test_analysis.md").write_text("\n".join(rows), encoding="utf-8")

    ok = sum(1 for r in records if r.get("parse") == "ok")
    print(f"\n{ok}/{len(records)} parsed. results/parser_test_analysis.md 확인.")


if __name__ == "__main__":
    main()
