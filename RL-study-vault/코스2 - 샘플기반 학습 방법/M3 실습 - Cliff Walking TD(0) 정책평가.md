---
title: "M3 실습 - Cliff Walking TD(0) 정책평가"
course: "코스2 - 샘플기반 학습 방법"
module: "Module 3 실습"
type: 실습
tags:
  - rl/코스2-샘플기반
  - 유형/실습
  - 개념/TD-0
  - 개념/정책평가
  - 개념/환경설계
  - 개념/절벽걷기
---


# Cliff Walking TD(0) Policy Evaluation — 실습 회고

> **Course**: Reinforcement Learning Specialization, Course 2 (Sample-based Learning Methods), Module 2
> **Environment**: Cliff Walking (4×12 grid)
> **Algorithm**: TD(0) — On-line tabular policy evaluation
> **Task**: 환경 동역학 $p$를 모르는 상황에서, 주어진 정책 $\pi$의 가치함수 $V(s)$를 샘플만으로 추정

---

## 📌 TL;DR

- MDP를 **코드로 옮기는 작업**이 알고리즘 구현보다 더 많이 배우게 했다.
- 환경 설계의 **한 줄**(terminal 처리, reward 비율)이 학습 결과의 형태를 결정한다.
- Bootstrap의 위태로움은 terminal anchor가 잡아준다.
- 같은 환경에서도 **정책이 다르면 가치함수가 완전히 다르다** — 당연하지만 직접 봐야 체감됨.
- Stochastic policy 실험이 가장 흥미로웠다: **risk가 가치에 자동으로 내재화되는 메커니즘**을 직접 관찰.
- 단, 이번 실습은 tabular라 깔끔하게 수렴했지만, **시뮬레이션 + 함수 근사**에서는 같은 깔끔함을 기대하면 안 됨.

---

## 1. 이 실습이 뭐였는지

핵심은 두 가지였다.

1. **MDP를 코드로 옮기는 작업** — `env_init`, `env_step`, state encoding
2. **Bootstrap 기반 가치 추정** — `agent_step`의 한 줄

알고리즘보다 환경 설계 쪽에서 얻은 게 더 많았다.

---

## 2. 환경 구현하면서 든 고민들

### 2.1 State encoding — 왜 굳이 1차원으로?
![[c2m3lab-image.png]]

처음에 좌표를 인덱스로 펴는 게 어색했다. $(x, y)$ 그대로 쓰면 안 되나?

```python
def state(self, loc):
    x, y = loc
    return x * self.grid_w + y
```

강의 설명을 보면 의도가 명확하다:

> "에이전트는 환경의 공간적 구조를 알 필요 없다. 그냥 상태 번호 → 가치 테이블 lookup만 하면 된다."

이게 6주차 LEO 위성 논문에서 정리했던 **첫 번째 MDP 설계 원칙**(상태에는 의사결정에 필요한 정보만)과 정확히 같은 얘기다. **공간 구조는 환경이 알고 있으면 되고, 에이전트는 추상화된 상태만 받으면 된다.**

> 💡 다만 이건 tabular니까 가능한 거고, 시뮬레이션 환경(Isaac Lab)에서는 상태가 연속이고 차원도 크다. 거기선 "공간 정보를 펼친다"가 아니라 **"공간 정보를 함수 근사로 압축한다"** 로 형태가 바뀐다. 핵심 정신은 같음: 에이전트가 봐야 할 것만 보여준다.

---

### 2.2 Cliff 처리 — `terminal=False`가 핵심

```python
if self.agent_loc in self.cliff:
    reward = -100
    terminal = False              # ← 여기
    self.agent_loc = self.start_loc
```

처음엔 절벽에 빠지면 episode가 끝나는 게 자연스러워 보였는데, 코드를 보니 **일부러 안 끝낸다.** 시작점으로 되돌리고 episode 안에서 계속 굴린다.

학습 신호 측면에서 굉장히 다르다:

| 설계 선택 | 학습 신호 밀도 |
|---|---|
| `terminal=True`로 끝냄 | 한 episode = 한 번의 절벽 신호 |
| `terminal=False`로 reset | 한 episode 안에서 절벽 빠지고 다시 빠지고 반복 |

**종료 조건 한 줄이 학습 속도를 결정한다.** 이게 코드 한 줄로 환경 설계자가 의도적으로 만들 수 있다는 게 인상적이었다.

> 💡 이 직관은 Isaac Lab에서도 그대로 쓸 수 있다. 로봇이 장애물에 충돌했을 때 episode를 끝낼지, 페널티만 주고 reset해서 계속 굴릴지에 따라 학습 효율이 달라질 것. **시뮬레이션에서는 reset 비용이 거의 0이니까 이 선택지가 더 풍부하다.**

---

### 2.3 경계 처리 — 모듈화의 이유

```python
def isInBounds(x, y, width, height):
    return 0 <= x < height and 0 <= y < width
```

벽에 부딪히면 제자리. 별거 아닌데 **별도 함수로 분리**한 게 좋았다. **물리 제약(어디까지 갈 수 있나) ≠ 행동 의미(어느 방향으로 가려고 했나)** 라는 분리가 명시적으로 된다.

> 💡 시뮬레이션에서는 이게 더 중요해진다. 로봇이 벽을 통과할 수는 없고, 물리 엔진이 충돌 처리를 알아서 해준다. 그런데 reward 설계에서 "벽에 부딪혔다"는 사실 자체를 페널티로 줄지 말지는 별개 결정. **물리 제약과 학습 신호를 분리해서 생각하는 습관**이 필요하다.

---

## 3. TD(0) 알고리즘 — 차이는 한 줄에 있다

```python
# agent_step (episode 계속)
td_target = reward + self.discount * self.values[state]

# agent_end (terminal)
td_target = reward                      # γV(s') 없음
```

수식적으로는 알고 있었는데, 코드로 보니까 **벨만 방정식의 base case가 어디서 끊어지는지가 한 줄로 드러난다.**

5주차에 정리했던 벨만 방정식의 재귀 구조:

$$V(s) = R + \gamma V(s')$$

이게 terminal에서 $V(s) = R$로 끊어지는데, 이 끊김이 `agent_end`의 한 줄 차이로 구현된다. **이론과 코드가 1:1로 대응되는 순간**이라 좋았다.

---

### 3.1 추정치를 추정치로 갱신한다는 게 위태로워 보이는데 왜 수렴할까

TD(0)의 핵심 업데이트:

$$V(S_t) \leftarrow V(S_t) + \alpha\left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

여기서 $V(S_{t+1})$도 결국 추정치다. **추정치로 추정치를 갱신**한다. 처음엔 이게 발산할 것 같았는데, 실험에서 5000 episode 돌리면 RMSVE가 깔끔하게 0으로 점근 수렴한다.

이유:

- Terminal에서는 추정치가 아닌 **실제 reward**가 들어옴 (anchor)
- 이 anchor가 매 episode 끝마다 들어와서, 추정치 chain이 결국 ground truth로 끌려감

Monte Carlo와의 결정적 차이는 학습 빈도다:

| 방식 | 학습 시점 |
|---|---|
| Monte Carlo | Episode가 끝나야 한 번 학습 |
| TD(0) | 매 step마다 학습 |

> ⚠️ 단, **anchor가 정기적으로 들어와야** 수렴한다. Episode가 너무 길거나 terminal에 도달하기 너무 어려우면 TD는 학습이 느려질 수 있다. → 시뮬레이션의 **reward sparsity 문제**로 연결됨.

---

## 4. 세 정책 실험에서 본 것

**공통 설정**: $\gamma=1$, $\alpha=0.01$, 5000 episodes, seed=0

> 📊 **[그래프 1: Optimal Policy 가치함수 + RMSVE 수렴 곡선]**
![[c2m3lab-image-1.png]]
> 📊 **[그래프 2: Safe Policy 가치함수]**
![[c2m3lab-image-2.png]]
> 📊 **[그래프 3: Near-Optimal Stochastic Policy 가치함수]**
![[c2m3lab-image-3.png]]
---

### 4.1 Optimal vs Safe — 같은 환경, 다른 정책, 다른 가치

| 정책 | 경로 | Step 수 | 가치 특성 |
|---|---|---|---|
| Optimal | 절벽 바로 위 최단 경로 | 13 | 모든 상태 가치 큼 (0에 가까움) |
| Safe | 벽 따라 우회 | 17 | 가치 약간 낮지만 안정적 |

**환경은 똑같은데 가치함수의 형태가 완전히 다르다.**

당연한 얘기지만 직접 보니까 다르게 느껴졌다. **"환경의 가치"가 아니라 "정책의 가치"** 라는 걸 시각적으로 체감함. Policy evaluation이 왜 **prediction** 문제라고 불리는지, 그리고 왜 **control**(최적 정책 찾기)과 분리되는 개념인지가 명확해졌다.

---

### 4.2 Near-Optimal Stochastic — 가장 흥미로웠던 실험

```python
policy[36] = [0.9, 0.1/3., 0.1/3., 0.1/3.]   # 시작점에서 90% UP
for i in range(24, 35):
    policy[i] = [0.1/3., 0.1/3., 0.1/3., 0.9] # 윗줄 90% RIGHT
policy[35] = [0.1/3., 0.1/3., 0.9, 0.1/3.]   # 우상단에서 90% DOWN
```

Optimal과 거의 같은 정책인데 10%만 무작위. **이 10%가 절벽 근처 상태의 가치를 눈에 띄게 떨어뜨린다.**

왜 흥미로웠냐면, **risk가 가치함수에 자동으로 내재화되는 메커니즘**을 본 거다.

- 절벽 페널티: $-100$
- 일반 step cost: $-1$
- 비율: **100배**

이 비율 때문에, 단 10%의 절벽 추락 확률만으로도 절벽 근처 상태의 가치가 크게 깎인다.

---

### 4.3 SARSA vs Q-learning의 분기점을 미리 본 셈

이 구조 그대로 다음 Module의 control 알고리즘이 동작한다:

- **SARSA (on-policy)** → 자기가 실제로 행동한 결과를 가치에 반영 → 절벽 근처가 위험하다는 걸 가치함수가 알게 됨 → Safe path 선호
- **Q-learning (off-policy)** → max로만 가치를 추정 → 절벽 위 위험은 가치에 안 들어감 → Optimal path 선호 → 실제 실행 시 ε-greedy로 추락

이번 실습은 prediction까지인데, **왜 SARSA와 Q-learning이 Cliff Walking에서 다른 정책으로 수렴하는지의 근본 원인을 prediction 레벨에서 미리 본 셈.** 이게 가장 큰 수확.

---

## 5. 캡스톤(Isaac Lab 시뮬레이션)으로 가져갈 직관들

코드 단위 실습이지만, 시뮬레이션 기반 실험으로 옮길 때 직접 쓸 수 있을 직관들.

### 5.1 Reward 비율이 정책 성격을 결정한다

Cliff Walking에서 본 직관:

- 절벽 $-100$, step $-1$의 **100배 차이**가 "위험을 가치에 반영시키는 강도"를 결정

Isaac Lab navigation에 그대로 적용:

- 충돌 페널티 vs 도달 보상 vs step cost의 **비율**
- 이 비율이 로봇이 *안전하지만 느린 경로* vs *빠르지만 위험한 경로* 중 뭘 선호할지 결정

> 🎯 같은 환경에 **reward scale만 바꿔서 정책이 어떻게 달라지는지** 비교하는 게 캡스톤의 핵심 contribution이 될 수 있을 것 같다.

---

### 5.2 Terminal vs reset의 선택지

Cliff에서 본 "terminal=False + reset" 패턴은 시뮬에서 더 유용할 듯:

| 설계 | 신호 밀도 |
|---|---|
| 로봇 장애물 충돌 → episode 끝 | 희소함 |
| 로봇 장애물 충돌 → 큰 페널티 + 안전한 위치로 리셋 | 밀도 높음 |

시뮬레이션은 reset이 거의 무료라서 후자가 훨씬 효율적일 가능성이 높다. **이건 실험으로 확인해볼 만한 지점.**

---

### 5.3 Stochasticity가 정책 평가에 미치는 영향 = Sim-to-Real에 대한 힌트

Near-Optimal Stochastic 실험에서 본 게 사실 **sim-to-real gap의 작은 버전**이다.

실제 로봇은 결정론적으로 안 움직인다:

- 액추에이터 노이즈
- 센서 노이즈
- 외란

**이 stochasticity가 가치함수에 어떻게 내재화되는가가 안전성을 결정한다.**

> 🎯 캡스톤에서 **domain randomization**을 적용한다면, 사실은 "환경에 의도적인 stochasticity를 주입해서 정책이 risk를 가치에 내재화하도록 강제하는 것"임. 이번 실습이 그 메커니즘을 작은 사례로 보여줬다.

---

### 5.4 ⚠️ Tabular와 함수 근사는 다르다 — 경계해야 할 점

이번에 본 수렴 보장(RMSVE → 0)은 **tabular니까 가능한 거다.** 함수 근사(특히 신경망)에서는:

> **Deadly Triad** = Off-policy + Bootstrap + Function Approximation
> → 이론적 수렴 보장이 깨짐
> → 실제 학습이 발산하거나 진동할 수 있음

Isaac Lab에서 PPO 같은 알고리즘을 쓰는 이유 중 하나가 이거다. **On-policy로 deadly triad를 피하면서, 신경망의 표현력은 유지.**

**이번 실습의 수렴 경험을 "원래 RL은 잘 수렴한다"로 일반화하면 안 된다.** 시뮬에서는 안정화에 훨씬 더 많은 노력이 들어갈 것.

---

## 6. 다음으로

- **Module 3-4 (TD Control: SARSA, Q-learning)**: prediction → control 넘어가서, 이번에 본 가치함수가 실제로 정책을 만들어내는 과정 학습
- **Course 3 (Function Approximation)**: tabular의 한계를 넘어, 신경망 기반 가치 함수 학습
- 그 후 **Isaac Lab example task + reward shaping** 캡스톤 미니 프로토타입

---




---

## 🔗 관련 노트
- [[M3 - TD 학습(예측)]]
- [[절벽걷기 - SARSA vs Q러닝 비교]]
