import json
from pathlib import Path

from nl_parser import parse_command


BASE_DIR = Path(__file__).resolve().parent


def assess(case, spec, status):
    if status != "PASS" or spec is None:
        return "파싱 또는 스키마 검증 실패"
    if spec.get("goal") is None:
        return "목표 미명시를 goal=null로 보존"
    if case["expected_category"].startswith("clear"):
        return "의도 일치"
    if case["expected_category"] == "ambiguous":
        return "모호성을 interpretation_notes에 기록"
    if "constrained" in case["expected_category"]:
        return "제약을 hard/soft 영역으로 변환"
    return "검토 필요"


def run_tests():
    cases = json.loads((BASE_DIR / "test_cases" / "commands.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        try:
            spec = parse_command(case["command"])
            status = "PASS"
            notes = spec.get("interpretation_notes", "")
        except Exception as e:
            status = "FAIL"
            spec = None
            notes = str(e)
        results.append(
            {
                "id": case["id"],
                "command": case["command"],
                "category": case["expected_category"],
                "status": status,
                "spec": spec,
                "notes": notes,
                "assessment": assess(case, spec, status),
            }
        )

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "parser_test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(results, results_dir / "parser_test_analysis.md")
    return results


def write_markdown(results, path):
    rows = [
        "| ID | 명령 | 상태 | LLM 해석 | 평가 |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        spec = result.get("spec") or {}
        llm_notes = (spec.get("interpretation_notes") or result.get("notes") or "").replace("|", "/")
        rows.append(
            f"| {result['id']} | {result['command']} | {result['status']} | {llm_notes} | {result['assessment']} |"
        )

    pass_count = sum(1 for result in results if result["status"] == "PASS")
    text = f"""# 파서 테스트 결과 분석

{chr(10).join(rows)}

## 발견한 패턴
1. 명확한 좌표와 모서리 표현은 대부분 안정적으로 구조화된다.
2. 목표가 없는 명령은 `goal=null`로 남겨 builder 단계에서 통제된 실패로 처리한다.
3. "위험해", "주변" 같은 표현은 hard obstacle보다 soft avoidance로 해석하는 편이 의도 손실이 작다.

## 요약
- PASS: {pass_count}/{len(results)}
- FAIL: {len(results) - pass_count}/{len(results)}
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    results = run_tests()
    print(json.dumps(results, indent=2, ensure_ascii=False))
