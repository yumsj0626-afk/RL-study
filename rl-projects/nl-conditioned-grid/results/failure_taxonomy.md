# NL to MDP 변환 시스템의 실패 지점 분류

## 실험별 결과 요약

| 실험 | 상태 | 주요 결과 | 가설 판정 |
|---|---|---|---|
| A_clear | success | `(7,7)`, `preference=shortest`; 평균 경로 길이 `14.0`, 성공률 `1.0` | 충족 |
| B_constrained | success | 4개 hard obstacle로 변환; obstacle 진입 `0`, 성공률 `1.0` | 충족 |
| C_ambiguous | success | `preference=safe`로 해석됐지만 `soft_avoid=[]`; 평균 경로 길이 `14.0` | 부분 충족 |
| D_failure | build_failed | `goal=null`; builder가 `Goal is underspecified`로 중단 | 충족 |

## 분류 체계

| 실패 유형 | 발생 위치 | 이번 실행에서의 관측 | 대응 방향 |
|---|---|---|---|
| 파싱 실패 | LLM 단계 | 0건. OpenAI 호출과 JSON/schema validation은 모두 통과했다. | API 오류와 schema 오류를 계속 별도 로깅한다. |
| 목표 미명시 | LLM/Builder 단계 | 1건. D_failure에서 `goal=null`이 builder 실패로 이어졌다. | 사용자 clarification loop: "어디로 갈까요?" |
| 의도 약화 | LLM/spec 단계 | 1건. C_ambiguous의 "안전하게"는 `safe`로 보존됐지만 위험 대상이 없어 reward 효과가 없었다. | `safe`의 기본 의미를 margin, 속도, unknown-zone 회피 등으로 구체화한다. |
| hard/soft 제약 분류 | LLM 단계 | B_constrained는 "절대 피해"를 hard obstacle로 올바르게 분류했다. | "가급적", "위험", "막힘" 표현별 regression test를 늘린다. |
| 학습 실패 | RL 단계 | 0건. 성공 실험 3개 모두 성공률 `1.0`, timeout `0`. | 더 큰 grid와 sparse reward에서 별도 검증한다. |

## 빈도 통계

```json
{
  "success": 3,
  "build_failed": 1,
  "parse_failed": 0,
  "learning_failed": 0
}
```

## 가장 위험한 실패 유형

가장 중요한 리스크는 **목표 미명시와 의도 약화**다. D_failure는 시스템이 멈춰서 재질문할 수 있으므로 비교적 안전한 실패다. 반면 C_ambiguous처럼 "안전하게"가 `preference=safe`로 기록됐지만 실제 환경에는 위험 대상이 없는 경우, 겉으로는 성공처럼 보이면서 사용자의 의도가 약하게 반영된다.

## Capstone Bridge에서의 의미

캡스톤으로 확장할 때 가장 먼저 해결해야 할 문제는 "명령의 형식적 성공"과 "사용자 의도의 실제 반영"을 분리해서 평가하는 것이다. grid에서는 `safe`가 비어 있어도 목표 도달이 가능하지만, 로봇 환경에서는 안전이 속도, 거리, 충돌 margin, 사람/장애물 주변 비용으로 구체화되어야 한다.

따라서 v2 스키마는 `preference`만 두기보다 `safety_constraints`, `minimum_clearance`, `forbidden_regions`, `ask_user_if_missing_goal` 같은 필드를 추가하는 편이 좋다. 이번 정찰의 재사용 가능한 자산은 LLM parser, schema validation, builder failure logging이고, 그대로 쓰기 어려운 부분은 tabular state/action 표현과 단순 reward shaping이다.
