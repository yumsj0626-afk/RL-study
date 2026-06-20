---
title: "M2 퀴즈 - MC·오프폴리시 복습"
course: "코스2 - 샘플기반 학습 방법"
module: "Module 2"
type: 퀴즈
tags:
  - rl/코스2-샘플기반
  - 유형/퀴즈
  - 개념/몬테카를로
  - 개념/off-policy
  - 개념/coverage
  - 개념/중요도샘플링
---


![[c2m2quiz-image.png]]

# Monte Carlo & Off-policy RL — 오답노트 & 해설지

---

## Q1. Monte Carlo를 적용할 수 있는 조건

**핵심 조건: episodic task여야 한다**

MC는 return G_t를 계산하기 위해 에피소드가 끝날 때까지의 전체 보상 시퀀스가 필요하다.
Continuing task는 종료 시점이 없으므로 return 자체를 정의할 수 없어 MC 적용 불가.

데이터 소스(배치 데이터 vs 모델 샘플)는 MC 적용 가능 여부에 영향을 주지 않는다.
에피소드 단위로 완결되기만 하면 어디서 온 데이터든 상관없다.

> **판단 기준 한 줄**: "에피소드가 끝나는 문제인가?" → Yes면 MC 가능.

---

## Q2. Off-policy learning의 정의와 예시

**핵심 정의: behavior policy ≠ target policy**

- **behavior policy**: 실제로 행동하며 데이터를 생성하는 정책
- **target policy**: 학습하고자 하는 정책

두 정책이 다르면 off-policy. 데이터를 생성한 주체와 학습 대상이 다른 모든 상황이 해당된다.

| 상황 | behavior | target | off-policy? |
|---|---|---|---|
| ε-greedy로 행동하며 최적 정책 학습 | ε-greedy | greedy 최적 | ✅ |
| 인간 전문가 데이터로 에이전트 학습 | 인간 전문가 | 에이전트 정책 | ✅ |
| SARSA (자기 정책으로 학습) | π | π (동일) | ❌ on-policy |

> Q-learning이 off-policy의 대표 사례: ε-greedy로 행동하지만 업데이트는 greedy 기준으로 함.

---

## Q3 & Q10. Coverage 조건 (off-policy MC의 필수 요건)

**조건: π(a|s) > 0 이면 반드시 b(a|s) > 0**

off-policy MC는 importance sampling으로 작동한다:

$$\rho = \frac{\pi(a|s)}{b(a|s)}$$

b(a|s)가 분모에 있으므로, target policy π가 선택할 수 있는 행동은 behavior policy b도 반드시 경험한 적 있어야 한다. 아니면 분모가 0이 되어 비율 자체가 정의 불가능해진다.

**방향 혼동 주의**
- π > 0 → b > 0 ✅ (correct)
- b > 0 → π > 0 ❌ (wrong, 방향이 반대)

b는 π보다 더 넓게 탐험해도 되지만, 더 좁으면 안 된다.

> **직관**: "target이 가는 곳은 behavior도 가본 적 있어야 한다."

---

## Q4. Greedy policy 결정 가능 조건 — v vs q

**v(s) 기반 greedy:**

$$\pi'(s) = \arg\max_a \sum_{s'} p(s'|s,a)[r + \gamma v(s')]$$

전이 확률 p(s'|s,a)가 필요 → **모델 필수**. 모델 없으면 다음 상태를 모르므로 행동별 가치 비교 불가.

**q(s,a) 기반 greedy:**

$$\pi'(s) = \arg\max_a\ q(s,a)$$

그냥 argmax → **모델 불필요**. q가 이미 행동까지 인덱싱되어 있으므로 바로 비교 가능.

| 조건 | greedy 결정 가능? |
|---|---|
| v + 모델 있음 | ✅ |
| v + 모델 없음 | ❌ |
| q + 모델 있음 | ✅ |
| q + 모델 없음 | ✅ |

> **q를 쓰면 model-free 가능 → 실용 RL에서 q를 선호하는 근본 이유.**

---

## Q5. MC prediction의 첫 업데이트 시점

**첫 에피소드가 끝난 후**

return G_t는 에피소드 종료 시점까지의 전체 보상이 모여야 계산 가능하다.
에피소드 도중에는 미래 보상을 알 수 없으므로 업데이트 불가.

- TD: 매 스텝마다 업데이트 가능 (bootstrap 사용)
- MC: 에피소드 종료 후에야 첫 업데이트 가능

---

## Q6. 에피소드 종료 시 업데이트 횟수의 결정 요인

**에피소드 길이에 의존**

MC 알고리즘의 innermost loop은 에피소드에서 방문한 각 타임스텝 t를 순회한다:

---
에피소드가 길수록 방문한 state가 많아지고 → 업데이트 횟수도 비례해서 증가.
State 수나 action 수는 이 루프의 반복 횟수와 무관하다.

---

## Q7. MC prediction이 하는 일

**Sample returns를 평균낸다**

$$V(s) \leftarrow \text{average of } G_t \text{ whenever } S_t = s$$

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

**혼동 포인트**
- reward(즉각 보상 한 스텝) ≠ return(에피소드 끝까지 할인 합산)
- MC는 return을 평균내는 것이지, reward를 평균내는 게 아님
- sweep(DP처럼 모든 state 순회)도 아님
- model planning도 아님 (model-free)

---

## Q8. MC 추정값 계산

상태 s를 returns 8, 4, 3으로 3번 방문한 경우:

$$V(s) = \frac{8 + 4 + 3}{3} = \frac{15}{3} = 5$$

단순 평균. 방문 횟수가 늘수록 추정값이 진짜 기댓값으로 수렴한다 (대수의 법칙).

---

## Q9. ε-greedy에서 최고값 행동의 선택 확률

$$P(\text{greedy}) = (1-\epsilon) \cdot 1 + \epsilon \cdot \frac{1}{A} = 1 - \epsilon + \frac{\epsilon}{A}$$

$$P(\text{non-greedy}) = \frac{\epsilon}{A}$$

**구조 이해**

ε-greedy는 두 가지를 섞는다:
- (1-ε) 확률로 → greedy 행동만 선택
- ε 확률로 → A개 행동 중 균등 랜덤 (greedy 포함)

greedy 행동은 탐험 구간(ε)에서도 뽑힐 수 있으므로 항상 non-greedy보다 확률이 높다.

**검산**: $(1 - \epsilon + \frac{\epsilon}{A}) + (A-1) \cdot \frac{\epsilon}{A} = 1$ ✅

---

## DP / MC / TD 핵심 비교

| 기준 | DP | MC | TD |
|---|---|---|---|
| 모델 필요? | ✅ 필요 | ❌ 불필요 | ❌ 불필요 |
| Episodic 필요? | ❌ | ✅ 필수 | ❌ |
| Bootstrap? | ✅ | ❌ | ✅ |
| 업데이트 시점 | 매 스텝 (sweep) | 에피소드 종료 후 | 매 스텝 |
| 무엇으로 업데이트 | Bellman 기댓값 | 실제 return G_t | 추정 return |
| 대표 알고리즘 | Policy Iteration | MC Control | Q-learning, SARSA |

---

## 혼동 포인트 정리

**reward vs return**
- reward = 한 스텝의 즉각 신호 $R_t$
- return = 에피소드 끝까지 할인 합산 $G_t = \sum \gamma^k R_{t+k+1}$
- MC는 return을 평균냄 (reward 평균이 아님)

**sweep vs visit**
- sweep = DP가 모든 state를 체계적으로 순회
- visit = MC가 에피소드를 따라가다 지나친 state
- 같은 개념이 아님. 문제에서 힌트로 명시할 정도로 자주 혼동됨

**coverage 방향**
- π > 0 → b > 0 ✅
- b > 0 → π > 0 ❌
- b가 π를 포함해야 하는 것이지, 반대가 아님

**v vs q + 모델**
- v만 있으면 greedy 불가 (모델 있어야 다음 상태 계산 가능)
- q만 있으면 greedy 가능 (행동까지 인덱싱되어 있으므로 argmax만 하면 됨)
- q가 model-free RL에서 핵심인 이유

**on-policy vs off-policy**
- SARSA: 자기 정책으로 행동하고 자기 정책을 업데이트 → on-policy
- Q-learning: ε-greedy로 행동하고 greedy 기준으로 업데이트 → off-policy


---

## 🔗 관련 노트
- [[M2 - 몬테카를로 방법]]
