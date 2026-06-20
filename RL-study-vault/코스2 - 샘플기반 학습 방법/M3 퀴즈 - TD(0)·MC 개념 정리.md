---
title: "M3 퀴즈 - TD(0)·MC 개념 정리"
course: "코스2 - 샘플기반 학습 방법"
module: "Module 3"
type: 퀴즈
tags:
  - rl/코스2-샘플기반
  - 유형/퀴즈
  - 개념/TD학습
  - 개념/부트스트래핑
  - 개념/bias-variance
---


![[c2m3quiz-image.png]]
# TD(0) & Monte Carlo — 퀴즈 개념 정리

> Course 2, Module 2
> 문제별 핵심 개념과 접근법 정리

---

## Q1. TD(0) is a solution method for?

### 알아야 할 개념
RL의 두 가지 핵심 문제 유형을 구분해야 한다.

| 문제 유형 | 목표 | 대표 알고리즘 |
|---|---|---|
| **Prediction** | 주어진 정책 π에 대해 V(s) 추정 | TD(0), MC |
| **Control** | 최적 정책 π* 탐색 | SARSA, Q-learning |

### 접근법
"TD(0)가 뭘 하는 알고리즘인가"를 먼저 물어봐야 한다.
TD(0)는 정책이 **주어진** 상태에서 그 정책의 가치함수를 추정하는 것 — 정책을 바꾸지 않는다.
정책을 바꾸는 건 control의 역할이다.

---

## Q2. Which methods use bootstrapping?

### 알아야 할 개념
> **Bootstrap** = 현재의 추정치를 이용해 다른 추정치를 갱신하는 것

```
TD(0):  V(S_t) ← V(S_t) + α[R + γV(S_{t+1}) - V(S_t)]
                                    ↑
                             이게 bootstrap — V(S_{t+1})은 추정치
                             
MC:     V(S_t) ← V(S_t) + α[G_t - V(S_t)]
                                ↑
                         G_t는 실제 return — 추정치 아님
```

### 접근법
"target에 추정치가 들어가 있는가?"를 보면 된다.
- $G_t$ → 실제 return → bootstrap ❌
- $V(S_{t+1})$ → 추정치 → bootstrap ✅

---

## Q3. DP vs TD — expected update vs sample update?

### 알아야 할 개념

| | DP | TD |
|---|---|---|
| 환경 모델 | 알고 있음 (전이확률 p) | 모름 |
| 업데이트 방식 | **Expected** — 가능한 모든 다음 상태를 확률 가중 평균 | **Sample** — 실제로 경험한 다음 상태 하나만 사용 |

### 접근법
"환경 모델을 알고 있는가?"가 핵심 질문이다.
- p를 알면 → 모든 경우를 계산 가능 → expected update
- p를 모르면 → 샘플로만 업데이트 → sample update

---

## Q4. 알고리즘과 수식 매칭

### 알아야 할 개념
두 수식의 차이는 **target**에 있다.

$$\text{MC target: } G_t \quad \text{(episode 끝까지의 실제 누적 보상)}$$

$$\text{TD target: } R_{t+1} + \gamma V(S_{t+1}) \quad \text{(한 step + bootstrap)}$$

### 접근법
수식에서 $G_t$가 보이면 MC, $V(S_{t+1})$이 보이면 TD(0).
나머지는 전부 같은 구조 $V \leftarrow V + \alpha[\text{target} - V]$ 다.

---

## Q5. TD와 MC — episodic/continuing task 적용 가능 여부

### 알아야 할 개념
MC는 $G_t$를 계산하려면 **episode가 반드시 끝나야** 한다.
TD는 매 step마다 업데이트하므로 **episode가 끝나지 않아도** 학습 가능.

| | Episodic | Continuing |
|---|---|---|
| MC | ✅ | ❌ |
| TD(0) | ✅ | ✅ |

### 접근법
"이 알고리즘이 언제 업데이트하는가?"를 생각하면 된다.
MC는 episode 종료가 전제 조건 → continuing task 불가.
TD는 step 단위 → 어떤 task든 가능.

---

## Q6. Terminal vs Non-terminal TD error

### 알아야 할 개념
TD error의 일반 수식:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

terminal 상태에서는 **미래가 없으므로** $V(S_{t+1}) = 0$ 으로 정의한다.
그러면 두 가지 동일한 표현이 가능하다:

```
방법 1: 일반 수식 그대로 쓰되 V(S_{t+1}) = 0 대입
        → δ_t = R_{t+1} + γ·0 - V(S_t) = R_{t+1} - V(S_t)

방법 2: 처음부터 bootstrap 항 제거
        → δ_t = R_{t+1} - V(S_t)
```

### 접근법
"다음 상태가 존재하는가?"를 먼저 확인한다.
- Non-terminal → $\gamma V(S_{t+1})$ 항 포함
- Terminal → $\gamma V(S_{t+1}) = 0$ → 항 제거

> 💡 실습 코드에서 `agent_step`과 `agent_end`가 나뉜 이유가 정확히 이것

---

## Q7. TD(0) 손계산

### 알아야 할 개념
α = 1일 때 업데이트 수식이 단순해진다:

$$V(S_t) \leftarrow V(S_t) + 1 \cdot [target - V(S_t)] = target$$

즉, **이전 추정치를 버리고 target으로 완전히 대체**한다.

### 접근법
trajectory를 순서대로 하나씩 쪼갠다.

```
trajectory: A, 0, B, 1, B, 0, T

step 1: S_t=A, R=0, S_{t+1}=B  →  V(A) ← 0 + 0.5×V(B) = 0 + 0.5 = 0.5
step 2: S_t=B, R=1, S_{t+1}=B  →  V(B) ← 1 + 0.5×V(B) = 1 + 0.5 = 1.5
step 3: S_t=B, R=0, terminal    →  V(B) ← 0  (bootstrap 없음)
```

주의: step 2에서 V(B)는 아직 1.0 (step 1에서 바뀐 건 V(A)만).
step 3에서 V(B)는 1.5 (step 2에서 갱신된 값).
**순서대로, 갱신된 값을 즉시 반영**해야 한다.

---

## Q8. TD(0) vs MC — Bias & Variance

### 알아야 할 개념

|  | TD(0) | MC |
|---|---|---|
| **Bias** | 있음 | 없음 (unbiased) |
| **Variance** | **낮음** | **높음** |

**왜 MC가 high variance인가?**
$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$
확률적 transition이 episode 전체에 걸쳐 누적 → variance가 쌓임

**왜 TD가 low variance인가?**
$R_{t+1} + \gamma V(S_{t+1})$
실제 randomness는 한 step $R_{t+1}$뿐, 나머지는 추정치로 대체 → variance 작음
대신 추정치를 쓰므로 bias 발생

### 접근법
Bias-Variance tradeoff로 접근하면 된다.
"얼마나 많은 실제 randomness를 사용하는가?" → 많을수록 unbiased지만 variance 높음.

---

## Q9. Batch MC vs Batch TD

### 알아야 할 개념
Batch 방식은 데이터를 반복적으로 학습에 활용한다.
수렴 시 아래를 만족하는 값으로 수렴한다:

- **Batch MC** → 관측된 return의 평균 (실제 경험에만 충실)
- **Batch TD** → MDP 구조의 기댓값 (전이 관계를 활용)

### 접근법

**MC**: 각 상태에서 시작한 에피소드의 return을 평균낸다.

```
V(B): return 목록 = 0, 1, 1, 1, 0, 0, 1, 0  →  평균 = 4/8 = 0.5
V(A): 에피소드 1에서만 등장, return = 0+0 = 0  →  V(A) = 0
```

**TD**: B의 기댓값을 먼저 구하고, A는 전이 관계로 역산한다.

```
V(B) = E[R] = 0.5  (B → terminal, reward 평균)
V(A) = E[R + γV(B)] = 0 + 1×0.5 = 0.5  (A → B, reward 0)
```

### 핵심 — V(A)에서 갈리는 이유

MC는 A를 방문한 에피소드가 딱 1개이고 그 return이 0이었다.
TD는 "A는 항상 B로 간다"는 **구조**를 활용해 V(B)=0.5를 반영한다.

> 💡 샘플이 적을수록 TD가 더 합리적인 추정을 내놓는 대표적인 사례

---

## Q10. TD(0)와 MC는 Markovian 환경에서 둘 다 수렴하는가?

### 알아야 할 개념

| 방법 | 수렴 조건 |
|---|---|
| MC | Markov 여부 무관하게 수렴 (unbiased이므로) |
| TD(0) | **Markovian 환경**에서만 수렴 보장 |

TD가 Markov property에 의존하는 이유: bootstrap에서 $V(S_{t+1})$을 쓰는데, 이게 의미 있으려면 $S_{t+1}$이 미래를 대표할 수 있어야 한다 — 이게 바로 Markov property다.

### 접근법
문제의 조건을 먼저 읽는다. **"given that the environment is Markovian"** 이 조건이 붙어 있으면 TD도 수렴 보장 → True.
이 조건이 없으면 TD는 수렴을 보장할 수 없으므로 False.

---

## Q11. TD(0) vs MC — online/offline

### 알아야 할 개념

| | 업데이트 시점 | 분류 |
|---|---|---|
| TD(0) | 매 step마다 즉시 | **Online** |
| MC | episode 완전히 끝난 후 | **Offline** |

**Online의 장점**: 긴 episode나 continuing task에서도 학습 가능. 실시간 적응 가능.
**Offline의 장점**: episode 전체 정보를 보고 업데이트하므로 안정적.

### 접근법
"$G_t$를 계산하려면 언제까지 기다려야 하는가?"
episode가 끝나야 안다 → offline.
한 step만 지나도 안다 → online.

---

## 📌 핵심 개념 총정리

```
                    MC          TD(0)         DP
───────────────────────────────────────────────────
Bootstrap           ❌           ✅            ✅
Update type       Sample       Sample       Expected
Update timing     Offline      Online          -
Bias               없음         있음          없음
Variance           높음         낮음          낮음
Continuing task    ❌           ✅             -
Model 필요         ❌           ❌            ✅
Markovian 수렴     ✅           ✅            ✅
```

### 문제 풀 때 체크리스트

1. **Bootstrap 쓰는가?** → target에 $V(S_{t+1})$ 있으면 YES
2. **언제 업데이트하는가?** → step 단위면 online, episode 끝이면 offline
3. **환경 모델 필요한가?** → p를 아는가
4. **Terminal인가?** → bootstrap 항 제거 여부 결정
5. **Markovian 조건이 붙어 있는가?** → TD 수렴 여부 결정



---

## 🔗 관련 노트
- [[M3 - TD 학습(예측)]]
