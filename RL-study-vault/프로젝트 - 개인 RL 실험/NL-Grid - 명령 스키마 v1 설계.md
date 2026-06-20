---
title: "NL-Grid - 명령 스키마 v1 설계"
course: "프로젝트 - 개인 RL 실험"
module: "nl-conditioned-grid"
type: 프로젝트
tags:
  - rl/프로젝트
  - 유형/프로젝트
  - 개념/스키마설계
  - 개념/MDP매핑
  - 개념/제약타입
  - 개념/언어grounding
---

# Command Schema v1 Design Rationale

| Natural language signal | MDP mapping | Rationale |
|---|---|---|
| Destination such as "go to the bottom-right corner" | `goal` and terminal reward | The destination defines the task objective directly. |
| Strong avoidance such as "never enter" or "blocked" | `obstacles` as hard environment dynamics | Hard constraints change the transition outcome through reset and large penalty. |
| Softer avoidance such as "prefer avoiding" or "dangerous" | `soft_avoid` as reward penalty | The cell remains passable, but the reward makes it unattractive. |
| Preference such as "fastest" or "safely" | `preference` reward shaping | The objective remains the same while the learning signal is biased toward speed or safety. |

`goal` allows `null` because the prompt explicitly asks the parser to report underspecified commands instead of inventing destinations. The builder treats `null` as a controlled failure.



---

## 🔗 관련 노트
- [[NL-Grid - 시스템 아키텍처]]
- [[NL-Grid - 정찰 정리]]
- [[실험 - 실패 분류·위험도 분석]]
