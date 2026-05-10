# NL-Conditioned Q-Learning: 자연어 명령 정찰

## 프로젝트 정체성

이 프로젝트는 캡스톤 자체가 아니라, 자연어를 MDP로 바꾸는 작은 파이프라인이 어디서 작동하고 어디서 깨지는지 확인하는 정찰 작업이다. LLM은 정책을 만들지 않고 자연어를 JSON spec으로 변환하며, 실제 행동 결정은 tabular Q-learning이 맡는다.

## 시스템 구조

1. `nl_parser.py`: OpenAI `gpt-4o`, `temperature=0`으로 자연어 명령을 command schema v1 JSON으로 변환한다.
2. `env_builder.py`: JSON spec을 검증하고 `NavGridEnv`로 변환한다.
3. `agent.py`: `(row, col)` 상태를 flat index로 바꿔 Q-learning을 수행한다.
4. `run_experiments.py`: 4개 실험을 실행하고 spec, policy plot, learning curve, interpretation, failure taxonomy를 저장한다.

## 실험 결과 요약

| 실험 | 명령 | 상태 | 핵심 결과 |
|---|---|---|---|
| A_clear | 오른쪽 아래 구석으로 가고 가장 짧게 가 | success | 평균 경로 길이 `14.0`, 성공률 `1.0`, 최단 경로 학습 |
| B_constrained | 중앙 2x2 영역은 절대 피해 | success | hard obstacle 4개, obstacle 진입 `0`, 성공률 `1.0` |
| C_ambiguous | 안전하게 (7,7)까지 | success | `safe`로 파싱됐지만 `soft_avoid=[]`라 안전 선호 효과는 제한적 |
| D_failure | 빠르게 가 | build_failed | `goal=null`, `Goal is underspecified`로 controlled failure |

## 핵심 발견 3가지

### 1. 명확한 좌표와 hard constraint는 잘 작동한다

A_clear와 B_constrained는 자연어가 명확할 때 파서, builder, Q-learning이 끝까지 연결됨을 보여준다. 특히 B_constrained에서는 "절대 피해"가 hard obstacle로 들어갔고, 최종 greedy rollout에서 obstacle 진입이 `0`회였다.

### 2. 목표 미명시는 안전하게 실패해야 한다

D_failure에서 LLM은 "빠르게"를 `preference=shortest`로 해석했지만 목적지를 만들지 않고 `goal=null`을 반환했다. builder는 `Goal is underspecified`로 중단했다. 이 동작은 hallucinated goal보다 낫고, 다음 단계에서는 사용자 재질문으로 이어져야 한다.

### 3. "안전하게"는 단독 preference로는 약하다

C_ambiguous는 파싱 자체는 성공했지만 `soft_avoid`가 비어 있어 `safe`가 실제 reward에 거의 영향을 주지 않았다. 즉, 성공률 `1.0`이더라도 사용자의 안전 의도가 충분히 구현됐다고 볼 수 없다.

## From NL-Grid to Capstone

| 차원 | 현재 정찰 | 캡스톤 예상 |
|---|---|---|
| State | `(row, col)` | 로봇 pose, velocity, lidar/camera 요약, 주변 객체 |
| Action | 4방향 이산 이동 | 연속 속도 명령 또는 waypoint/action primitive |
| Q 표현 | tabular Q-table | DQN, PPO, SAC 등 함수 근사 |
| 환경 | 8x8 deterministic grid | PyBullet/Gazebo/Isaac Sim 기반 동역학 환경 |
| 명령 스키마 | goal, obstacles, soft_avoid, preference | task goal, forbidden region, clearance, speed, temporal constraints |
| 실패 처리 | `goal=null`, builder `ValueError` | clarification loop, fallback planner, safety monitor |

### 재사용 가능한 자산

- LLM parser prompt와 schema validation 패턴
- 자연어 spec을 환경 생성자로 넘기는 builder 구조
- 실패 단계별 기록 방식: parse/build/learning
- 실험별 `interpretation.md` 자동 생성 흐름

### 확장 전에 해결할 문제

가장 중요한 문제는 **의도의 실효성 평가**다. grid에서는 "안전하게"가 `safe`로 파싱되면 성공처럼 보이지만, 실제 reward에 위험 대상이 없으면 정책은 달라지지 않는다. 캡스톤에서는 `safe`를 최소 거리, 속도 제한, 충돌 가능성, unknown area penalty 같은 구체적인 MDP 요소로 분해해야 한다.

또한 `goal=null` 실패는 재질문 루프가 필요하다. 자연어 명령이 "빠르게 가"처럼 목적지를 생략하면 시스템은 임의 목표를 만들지 말고 사용자에게 목적지를 물어야 한다.

## 실행 방법

```powershell
pip install -r ..\requirements.txt
$env:OPENAI_API_KEY = "sk-..."
python run_parser_tests.py
python run_experiments.py
```
