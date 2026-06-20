---
title: "NL-Grid - 프로젝트 일지"
course: "프로젝트 - 개인 RL 실험"
module: "nl-conditioned-grid"
type: 일지
tags:
  - rl/프로젝트
  - 유형/일지
  - 개념/일지
  - 개념/디버깅
---

## 2026-05-04 (작업 1회차)
- 완료한 단계: Phase 1 전체, Phase 2 schema/env/parser/builder/agent/pipeline.
- 발견한 것: 현재 PowerShell 세션에는 `OPENAI_API_KEY`가 없어 OpenAI parser 호출이 실패했다.
- 의미: 파이프라인은 실패 단계 기록까지 포함해야 하며, API 설정 실패는 가장 앞단에서 전체 실험을 막는다.
- 내일 할 것: API 키가 노출된 셸에서 `python run_parser_tests.py`와 `python run_experiments.py`를 재실행하고 성공 실험의 정책을 해석한다.



---

## 🔗 관련 노트
- [[NL-Grid - 정찰 정리]]
