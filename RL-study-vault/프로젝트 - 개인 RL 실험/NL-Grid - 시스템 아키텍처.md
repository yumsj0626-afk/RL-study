---
title: "NL-Grid - 시스템 아키텍처"
course: "프로젝트 - 개인 RL 실험"
module: "nl-conditioned-grid"
type: 프로젝트
tags:
  - rl/프로젝트
  - 유형/프로젝트
  - 개념/아키텍처
  - 개념/파이프라인
  - 개념/파서
  - 개념/환경빌더
---

# Architecture

## Pipeline

1. `nl_parser.py` loads `prompts/parser_prompt_v1.txt` and sends the command to OpenAI `gpt-4o` with `temperature=0`.
2. The returned JSON is validated against `schemas/command_schema_v1.json`.
3. `env_builder.py` converts the validated spec into `NavGridEnv`.
4. `agent.py` trains a tabular Q-learning policy on the generated environment.
5. `run_experiments.py` saves specs, plots, metrics, interpretations, and the failure taxonomy.

## Design Notes

- LLM output is treated as a parser result, not a policy.
- `goal=null` is allowed in the schema so underspecified commands become inspectable failures.
- Hard constraints are dynamics-level obstacles; soft constraints are reward-level penalties.
- Parser failure, builder failure, and learning failure are recorded separately.



---

## 🔗 관련 노트
- [[NL-Grid - 정찰 정리]]
- [[NL-Grid - 명령 스키마 v1 설계]]
