---
title: "M4 - TD 제어 종합 요약"
course: "코스2 - 샘플기반 학습 방법"
module: "Module 4"
type: 요약
tags:
  - rl/코스2-샘플기반
  - 유형/요약
  - 개념/SARSA
  - 개념/Q러닝
  - 개념/expected-sarsa
---


![[c2m4sum-image-8.png]]
# Module 4 — Temporal Difference Learning Methods for Control
## 통합 요약

---

## 전체 흐름 한눈에 보기

```
TD Prediction (Module 3)
    └─ V(s)를 샘플로 추정
    
TD Control (Module 4)
    ├─ Sarsa       : Q(s,a) 추정 + GPI → on-policy
    ├─ Q-learning  : Q*(s,a) 직접 학습 → off-policy
    └─ Expected Sarsa : 기댓값 직접 계산 → on/off-policy 모두 가능
```

Module 3에서 TD로 예측(prediction)을 다뤘다면, Module 4는 그것을 제어(control)로 확장합니다.
V(s) 대신 Q(s,a)를 학습해야 하는 이유는 단순합니다. 모델 없이 greedy 정책 개선을 하려면 행동들을 직접 비교할 수 있어야 하기 때문입니다.

---

## 1. Sarsa — on-policy TD Control

### 핵심 아이디어

GPI의 정책 평가 단계를 TD로 대체하고, 매 스텝마다 정책을 개선합니다.
이름은 업데이트에 사용하는 데이터 구조에서 왔습니다.

$$S_t,\ A_t,\ R_{t+1},\ S_{t+1},\ A_{t+1}$$

### 업데이트 수식

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

$A_{t+1}$을 업데이트 전에 behavior policy($\epsilon$-greedy)에서 미리 샘플링해야 합니다.
자신이 실제로 따르는 정책의 가치를 학습하는 **on-policy** 알고리즘입니다.

### 기반 방정식

벨만 방정식 (고정 정책 $\pi$ 하의 action value)

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t=s, A_t=a\right]$$

### Windy Gridworld 실험 결과

- 초반 에피소드: 수천 스텝 소요 → 점차 가속 → 약 7,000 스텝에서 최적 근방 수렴
- $\epsilon = 0.1$이므로 완전한 최적 정책은 아니지만 합리적 수준에 도달

> Monte Carlo는 종료 보장이 없는 환경(예: 시작 상태에서 왼쪽만 가는 정책)에서 학습 자체가 불가능합니다. Sarsa는 에피소드 중간에도 업데이트하기 때문에 나쁜 정책을 즉시 감지하고 벗어날 수 있습니다.

---

## 2. Q-learning — off-policy TD Control

### 핵심 아이디어

Sarsa와 달리 다음에 실제로 선택할 행동이 아니라, 다음 상태에서의 **최선 행동 가치**를 타겟으로 씁니다.

### 업데이트 수식

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

### 기반 방정식

벨만 **최적** 방정식

$$Q^*(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t=s, A_t=a\right]$$

Sarsa가 policy iteration의 샘플 버전이라면, Q-learning은 **value iteration의 샘플 버전**입니다.
정책 평가와 개선을 분리하지 않고 $Q^*$로 직접 수렴합니다.

### Importance Sampling 없이 off-policy가 가능한 이유

Q-learning의 target policy는 greedy입니다. Greedy 정책에서 비최대 행동의 선택 확률은 0이므로:

$$\mathbb{E}_\pi[Q(S', a)] = \max_a Q(S', a)$$

기댓값을 직접 계산할 수 있기 때문에 분포 보정이 필요 없습니다.

| 항목 | Sarsa | Q-learning |
| :--- | :--- | :--- |
| Target policy | $\epsilon$-greedy (behavior와 동일) | Greedy w.r.t. Q |
| Behavior policy | $\epsilon$-greedy | $\epsilon$-greedy (또는 임의) |
| 분류 | On-policy | Off-policy |
| 학습 대상 | $Q^\pi$ | $Q^*$ |

### Cliff World: Sarsa vs Q-learning

Q-learning은 최적 경로(절벽 바로 옆)를 학습하지만, $\epsilon$-greedy 탐험 중 자주 절벽에 떨어집니다.
Sarsa는 탐험 비용을 가치 추정에 반영해 더 안전한 우회 경로를 선택합니다.

> **이론적 최적 정책 ≠ 온라인 성능 최고**. 탐험 비용이 큰 환경에서는 on-policy 방법이 실제 누적 보상에서 앞설 수 있습니다.

---

## 3. Expected Sarsa — 가장 일반적인 형태

### 핵심 아이디어

Sarsa가 다음 행동을 샘플링해서 타겟을 만드는 대신, 에이전트가 이미 알고 있는 정책으로 **기댓값을 직접 계산**합니다.

### 업데이트 수식

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

### 세 알고리즘 타겟 비교

| 알고리즘 | 타겟 구성 방식 | 분산 |
| :--- | :--- | :--- |
| Sarsa | $Q(S', A')$ 샘플링 | 높음 |
| Expected Sarsa | $\sum_a \pi(a\|S') Q(S', a)$ 직접 계산 | 낮음 |
| Q-learning | $\max_a Q(S', a)$ | 가장 낮음 |

### Q-learning은 Expected Sarsa의 특수 케이스

Target policy가 greedy이면:

$$\sum_a \pi(a|S') Q(S', a) = \max_a Q(S', a)$$

즉, **Q-learning = Expected Sarsa (target policy: greedy)**

### Cliff World 실험 결과

- 거의 모든 $\alpha$ 범위에서 Sarsa보다 우수
- 결정론적 환경에서 업데이트 타겟 자체가 결정론적 → 큰 $\alpha$도 안정적으로 소화
- $\alpha$는 수렴 속도만 결정하고 최종값에 영향 없음
- Sarsa는 $\alpha$가 클수록 수렴 실패 가능

> Expected Sarsa의 강점은 자신의 정책에서 오는 무작위성을 직접 제거한다는 점입니다. 대신 행동 공간이 클수록 매 스텝 계산 비용이 선형으로 증가하는 트레이드오프가 있습니다.

---

## 최종 비교표

| 항목 | Sarsa | Q-learning | Expected Sarsa |
| :--- | :--- | :--- | :--- |
| 기반 방정식 | 벨만 방정식 | 벨만 최적 방정식 | 벨만 방정식 (기댓값) |
| 학습 대상 | $Q^\pi$ | $Q^*$ | $Q^\pi$ 또는 $Q^*$ |
| On/Off-policy | On | Off | 둘 다 가능 |
| 타겟 분산 | 높음 | 낮음 | 중간~낮음 |
| Importance Sampling | 불필요 | 불필요 | 불필요 |
| 계산 비용/스텝 | 낮음 | 낮음 | 행동 수에 비례 |
| $\alpha$ 민감도 | 높음 | 중간 | 낮음 |
| 포함 관계 | — | Expected Sarsa의 특수 케이스 | Sarsa + Q-learning 포괄 |



---

## 🔗 관련 노트
- [[M4 - TD 제어(SARSA·Q러닝)]]
