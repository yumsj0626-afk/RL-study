# Module 2 — On-policy Prediction with Approximation

> Course 3: Prediction and Control with Function Approximation
> 교재 범위: Sutton & Barto Ch. 9.1 – 9.4 (pp. 197–209)

지금까지(Course 1·2)의 모든 예측·제어 방법은 **표(table)** 기반이었다. 상태마다 칸이 하나씩 있고, 한 상태를 업데이트해도 다른 상태는 그대로였다. 이번 모듈부터는 상태 공간이 너무 커서(혹은 연속이라서) 표를 쓸 수 없는 경우를 다룬다. 핵심은 가치 함수를 **파라미터화된 함수(parameterized function)** $\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$ 로 표현하고, 가중치 벡터 $\mathbf{w} \in \mathbb{R}^d$ 를 학습하는 것이다.

여기서 보통 $d \ll |\mathcal{S}|$ 이므로, 하나의 가중치를 바꾸면 많은 상태의 추정값이 동시에 바뀐다. 이것이 **일반화(generalization)** 의 원천이자, 이번 파트 전체를 관통하는 새로운 어려움의 원천이다.

---

## 1. Moving to Parameterized Functions

표 기반 가치 함수는 상태 개수만큼의 항목을 갖는 거대한 룩업 테이블이다. 상태가 많아지면(예: 카메라 이미지는 우주의 원자 수보다 경우의 수가 많다) 표는 메모리·데이터 측면에서 불가능하다.

대안은 가치 함수를 가중치 벡터 $\mathbf{w}$ 로 매개되는 함수로 보는 것이다.

$$
\hat{v}(s, \mathbf{w}) \approx v_\pi(s)
$$

$\hat{v}$ 의 구체적 형태는 다양하다.

- 상태의 특징(feature)에 대한 **선형 함수** (이때 $\mathbf{w}$ 는 특징 가중치 벡터)
- **다층 인공 신경망** (이때 $\mathbf{w}$ 는 전 계층의 연결 가중치)
- **결정 트리** (이때 $\mathbf{w}$ 는 분할 지점과 잎 노드 값)

표는 이 틀의 특수한 경우로 볼 수 있다. 각 상태에 대응하는 항목이 곧 하나의 가중치이고, 상태별 특징 벡터가 one-hot 형태일 때가 정확히 표 기반이다.

> 부분 관측(partial observability) 연결: $\hat{v}$ 의 함수 형태가 상태의 특정 측면에 의존하지 못하게 설계되면, 그 측면은 관측되지 않는 것과 동일하게 작동한다. 함수 근사 결과들은 부분 관측 문제에도 그대로 적용된다. 다만 함수 근사가 **할 수 없는** 것은 과거 관측의 기억을 상태 표현에 추가하는 일이다.

---

## 2. Generalization and Discrimination

함수 근사를 평가하는 두 축을 구분해야 한다.

| 개념 | 의미 |
| :--- | :--- |
| 일반화 (Generalization) | 한 상태에 대한 업데이트가 유사한 다른 상태들의 추정값에 함께 영향을 미치는 정도 |
| 변별 (Discrimination) | 두 상태를 서로 다른 값으로 구분할 수 있는 능력 |

표 기반은 변별이 완벽하지만(모든 상태 독립) 일반화가 전혀 없다(데이터 효율이 나쁨). 함수 근사는 일반화를 얻는 대신 변별을 일부 희생한다.

좋은 함수 근사기는 이 둘의 균형을 잡는다. 너무 일반화하면 서로 다른 상태를 같은 값으로 뭉뚱그리고(변별 손실), 너무 변별하면 표와 다를 바 없어진다(일반화 손실). 이번 모듈 이후의 특징 설계(Module 3)는 결국 "어떤 상태들 사이에 일반화를 허용할 것인가"를 설계하는 작업이다.

---

## 3. Framing Value Estimation as Supervised Learning

각 업데이트를 $s \mapsto u$ 로 표기하자. $s$ 는 업데이트되는 상태, $u$ 는 그 추정값이 향해야 할 **목표(update target)** 다.

| 방법 | 업데이트 형태 $s \mapsto u$ |
| :--- | :--- |
| Monte Carlo | $S_t \mapsto G_t$ |
| TD(0) | $S_t \mapsto R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w}_t)$ |
| n-step TD | $S_t \mapsto G_{t:t+n}$ |
| DP | $s \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w}_t) \mid S_t = s]$ |

각 업데이트는 "입력 $s$, 출력 $u$" 라는 하나의 **훈련 예제(training example)** 로 해석할 수 있다. 즉 가치 추정을 **지도 학습(supervised learning)** 문제로 재구성하면, 기존의 함수 근사 기법(신경망, 회귀, 결정 트리 등)을 그대로 가져다 쓸 수 있다.

다만 RL에 그대로 쓰기엔 모든 지도 학습 기법이 적합하지는 않다. RL이 요구하는 추가 조건은 다음과 같다.

- **온라인 학습(online learning)**: 정적 데이터셋을 여러 번 훑는 방식이 아니라, 환경과 상호작용하며 점진적으로 들어오는 데이터로 학습해야 한다.
- **비정상 목표(nonstationary targets) 대응**: GPI에서 $\pi$ 가 변하면 목표 $q_\pi$ 도 변한다. 정책이 고정되어도 부트스트래핑(DP·TD) 목표는 $\mathbf{w}$ 에 의존하므로 시간에 따라 변한다.

---

## 4. The Value Error Objective ($\overline{VE}$)

표 기반에서는 학습된 값이 참값과 정확히 일치할 수 있어 "예측 품질"을 굳이 정의할 필요가 없었다. 하지만 함수 근사에서는 한 상태를 더 맞히면 다른 상태가 덜 맞는 트레이드오프가 불가피하므로, **어떤 상태의 오차를 더 중요하게 볼 것인지** 를 명시해야 한다.

이를 위해 상태 분포 $\mu(s) \ge 0,\ \sum_s \mu(s) = 1$ 을 도입한다. 그러면 자연스러운 목적 함수인 **평균 제곱 가치 오차(Mean Squared Value Error)** 가 정의된다.

$$
\overline{VE}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s)\,\big[\,v_\pi(s) - \hat{v}(s, \mathbf{w})\,\big]^2
$$

- $\mu(s)$ 는 보통 **상태 $s$ 에서 보내는 시간의 비율**로 선택된다. on-policy 학습에서 이는 **on-policy 분포(on-policy distribution)** 라 불린다.
- 연속 과제(continuing task)에서는 $\pi$ 하의 정상 분포(stationary distribution)다.
- 에피소드 과제에서는 시작 상태 분포 $h(s)$ 에 의존한다. 상태별 평균 방문 횟수 $\eta(s)$ 를 구한 뒤 정규화한다.

$$
\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar{s}) \sum_a \pi(a \mid \bar{s})\, p(s \mid \bar{s}, a)
\qquad
\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}
$$

> $\overline{VE}$ 가 RL의 "올바른" 목적인지는 사실 자명하지 않다. 우리의 궁극적 목적은 더 나은 정책을 찾는 것이고, $\overline{VE}$ 최소화가 곧 최선의 정책을 보장하지는 않는다. 그래도 현재로선 더 나은 대안이 명확하지 않아 $\overline{VE}$ 에 집중한다.

최적해는 전역 최적점 $\mathbf{w}^*$ ($\overline{VE}(\mathbf{w}^*) \le \overline{VE}(\mathbf{w})$ for all $\mathbf{w}$) 이지만, 신경망·결정 트리 같은 복잡한 근사기에서는 보통 지역 최적점(local optimum)에 도달하는 것이 최선이다. 일부 방법은 발산할 수도 있다(Ch. 11에서 다룸).

---

## 5. Introducing Gradient Descent

가중치 벡터를 매 예제마다 오차를 가장 빠르게 줄이는 방향으로 조금씩 조정하는 방법이 **확률적 경사 하강법(Stochastic Gradient Descent, SGD)** 이다.

참값 $v_\pi(S_t)$ 를 알고 있다고 가정할 때:

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \tfrac{1}{2}\alpha\, \nabla \big[\,v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\,\big]^2
= \mathbf{w}_t + \alpha\,\big[\,v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\,\big]\,\nabla \hat{v}(S_t, \mathbf{w}_t)
$$

여기서 $\nabla f(\mathbf{w})$ 는 $f$ 의 각 성분에 대한 편미분으로 이루어진 **기울기(gradient)** 벡터다.

$$
\nabla f(\mathbf{w}) \doteq \Big( \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_d} \Big)^\top
$$

**왜 한 번에 오차를 0으로 만들지 않고 작은 보폭($\alpha$)으로 가는가?** 우리는 어떤 상태도 오차가 0인 가치 함수를 기대하지 않는다. 우리가 원하는 것은 서로 다른 상태들의 오차를 **균형** 있게 맞추는 근사다. 한 예제를 한 번에 완전히 보정하면 그 균형이 깨진다. SGD의 수렴 보장은 $\alpha$ 가 시간에 따라 표준 확률 근사 조건(2.7)을 만족하며 감소할 것을 가정한다.

참값 대신 목표 $U_t$ 를 쓰는 일반형은 다음과 같다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\,\big[\,U_t - \hat{v}(S_t, \mathbf{w}_t)\,\big]\,\nabla \hat{v}(S_t, \mathbf{w}_t)
$$

$U_t$ 가 **불편 추정량(unbiased estimate)**, 즉 $\mathbb{E}[U_t \mid S_t = s] = v_\pi(s)$ 이면, 감소하는 $\alpha$ 하에서 지역 최적점으로 수렴이 보장된다.

---

## 6. Gradient Monte Carlo for Policy Evaluation

MC 목표 $U_t = G_t$ 는 정의상 $v_\pi(S_t)$ 의 **불편 추정량**이다. 따라서 위의 SGD 일반형에 그대로 대입하면 지역 최적해(선형 근사라면 전역 최적해)로의 수렴이 보장된다.

```
Gradient Monte Carlo Algorithm for Estimating v̂ ≈ v_π

Input: 평가할 정책 π
Input: 미분 가능한 함수 v̂ : S × R^d → R
Algorithm parameter: 스텝 사이즈 α > 0
가중치 w ∈ R^d 를 임의로 초기화 (예: w = 0)

Loop forever (각 에피소드마다):
    π 를 따라 에피소드 S_0, A_0, R_1, ..., R_T, S_T 생성
    Loop for each step t = 0, 1, ..., T-1:
        w ← w + α [G_t − v̂(S_t, w)] ∇v̂(S_t, w)
```

핵심은 목표 $G_t$ 가 $\mathbf{w}$ 와 **독립**이라는 점이다. 그래서 이것은 진정한(true) 경사 하강이다.

---

## 7. State Aggregation with Monte Carlo

**상태 집합(state aggregation)** 은 가장 단순한 일반화 함수 근사다. 상태들을 그룹으로 묶고, 각 그룹마다 추정값(가중치 성분) 하나를 둔다. 한 상태의 값은 그 그룹의 성분으로 추정되고, 업데이트 시 해당 그룹 성분만 갱신된다.

상태 집합은 SGD의 특수 케이스로, 기울기 $\nabla \hat{v}(S_t, \mathbf{w}_t)$ 가 $S_t$ 가 속한 그룹 성분에서는 1, 나머지에서는 0인 경우다.

**Example 9.1 — 1000-state Random Walk**

- 상태 1~1000을 좌→우로 번호 매김, 모든 에피소드는 중앙(state 500)에서 시작.
- 현재 상태에서 좌·우 인접 100개 상태 중 하나로 동일 확률 전이. 가장자리에서 빠지면 종료(좌측 종료 보상 $-1$, 우측 종료 보상 $+1$).
- 1000개 상태를 100개씩 10개 그룹으로 묶고, gradient MC + 상태 집합으로 100,000 에피소드 학습.

[그래프 위치 표시] — 참 가치 함수 $v_\pi$ vs. 근사값 $\hat{v}$ (계단 형태), 그리고 상태 분포 $\mu$

결과의 특징은 그룹 내에서 근사값이 상수이고 그룹 경계에서 급변하는 **계단 효과(staircase effect)** 다. 또한 각 그룹의 추정값은 그룹 내 참값의 단순 평균이 아니라, $\mu$ 가 큰 상태 쪽으로 **치우친다**. 예컨대 가장 왼쪽 그룹에서는 state 100이 state 1보다 3배 이상 자주 방문되므로, 그룹 추정값이 state 100의 참값 쪽으로 편향된다. → $\overline{VE}$ 의 가중치 $\mu$ 가 최종 해의 형태를 어떻게 바꾸는지를 시각적으로 보여주는 예시.

---

## 8. Semi-Gradient TD for Policy Evaluation

부트스트래핑 목표(예: TD(0)의 $U_t = R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w}_t)$)는 **현재 가중치 $\mathbf{w}_t$ 에 의존**한다. 따라서 편향되어 있고, SGD 유도의 핵심 단계(목표가 $\mathbf{w}_t$ 와 독립이라는 가정)가 무너진다.

이런 방법들은 가중치 변화가 추정값에 미치는 영향은 고려하지만 **목표에 미치는 영향은 무시**한다. 즉 기울기의 일부만 사용하므로 **준-경사법(semi-gradient method)** 이라 부른다.

```
Semi-gradient TD(0) for estimating v̂ ≈ v_π

Input: 평가할 정책 π
Input: 미분 가능한 함수 v̂ : S⁺ × R^d → R, 단 v̂(terminal, ·) = 0
Algorithm parameter: 스텝 사이즈 α > 0
가중치 w ∈ R^d 를 임의로 초기화 (예: w = 0)

Loop for each episode:
    S 초기화
    Loop for each step of episode:
        A ~ π(·|S) 선택
        행동 A 수행, R, S' 관측
        w ← w + α [R + γ v̂(S', w) − v̂(S, w)] ∇v̂(S, w)
        S ← S'
    until S is terminal
```

준-경사법의 트레이드오프:

- (단점) 경사법만큼 강건하게 수렴하지는 않는다.
- (장점) 학습이 **훨씬 빠르다** (Ch. 6·7의 결론과 동일).
- (장점) 에피소드 종료를 기다리지 않고 **연속적·온라인** 학습이 가능하다 → 연속 과제에 적용 가능, 계산상 이점.

---

## 9. Comparing TD and Monte Carlo with State Aggregation

**Example 9.2 — Bootstrapping on the 1000-state Random Walk**

상태 집합은 선형 함수 근사의 특수 케이스이므로, 같은 1000-state random walk에서 두 방법을 비교할 수 있다.

[그래프 위치 표시] — (왼쪽) semi-gradient TD(0)의 점근 가치 함수가 MC(Fig 9.1)보다 참값에서 더 멀다. (오른쪽) n-step semi-gradient TD의 성능 곡선이 표 기반 19-state random walk(Fig 7.2)와 놀랍도록 유사

요점: TD의 점근적 근사는 MC보다 **참값에서 더 멀지만**, 학습 속도(learning rate) 측면에서는 큰 잠재 이점을 유지한다. 어느 쪽이 나은지는 근사·문제의 성격과 학습을 얼마나 오래 지속하는지에 달려 있다 — 이는 9·10주차 Cliff Walking 실험에서 본 "이론적 최적성 ≠ 실제 성능"과 같은 결의 트레이드오프다.

n-step semi-gradient TD의 핵심 업데이트:

$$
\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha\,\big[\,G_{t:t+n} - \hat{v}(S_t, \mathbf{w}_{t+n-1})\,\big]\,\nabla \hat{v}(S_t, \mathbf{w}_{t+n-1})
$$

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{v}(S_{t+n}, \mathbf{w}_{t+n-1})
$$

---

## 10. The Linear TD Update (Linear Methods)

가장 중요한 특수 케이스는 $\hat{v}(\cdot, \mathbf{w})$ 가 가중치 $\mathbf{w}$ 의 **선형 함수**인 경우다. 각 상태 $s$ 에 특징 벡터 $\mathbf{x}(s) = (x_1(s), \dots, x_d(s))^\top$ 를 대응시키면:

$$
\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^{d} w_i\, x_i(s)
$$

각 특징 $x_i$ 는 선형 근사 함수 집합의 **기저 함수(basis function)** 다. 선형의 경우 기울기가 매우 단순해진다.

$$
\nabla \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)
$$

따라서 SGD 일반형이 다음으로 환원된다.

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\,\big[\,U_t - \hat{v}(S_t, \mathbf{w}_t)\,\big]\,\mathbf{x}(S_t)
$$

선형의 결정적 장점: **지역 최적점이 곧 전역 최적점**이다. (퇴화 케이스를 제외하면 최적점이 하나뿐) 따라서 gradient MC는 선형 근사에서 $\overline{VE}$ 의 전역 최적해로 수렴한다.

---

## 11. The True Objective for TD (TD Fixed Point)

선형 semi-gradient TD(0)도 수렴하지만, 그 수렴점은 $\overline{VE}$ 의 전역 최적점이 **아니라** 별도의 점이다. 연속 과제에서 업데이트는:

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\,\big(R_{t+1} + \gamma\,\mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t\big)\,\mathbf{x}_t
$$

정상 상태에서 기댓값을 취하면:

$$
\mathbb{E}[\mathbf{w}_{t+1} \mid \mathbf{w}_t] = \mathbf{w}_t + \alpha(\mathbf{b} - \mathbf{A}\mathbf{w}_t)
$$

$$
\mathbf{b} \doteq \mathbb{E}[R_{t+1}\mathbf{x}_t] \in \mathbb{R}^d,
\qquad
\mathbf{A} \doteq \mathbb{E}\big[\mathbf{x}_t(\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^\top\big] \in \mathbb{R}^d \times \mathbb{R}^d
$$

수렴한다면 다음을 만족하는 점, 즉 **TD 고정점(TD fixed point)** 에 도달한다.

$$
\mathbf{b} - \mathbf{A}\mathbf{w}_{TD} = 0 \quad\Rightarrow\quad \mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}
$$

**수렴 조건(직관):** 업데이트를 $\mathbb{E}[\mathbf{w}_{t+1} \mid \mathbf{w}_t] = (\mathbf{I} - \alpha\mathbf{A})\mathbf{w}_t + \alpha\mathbf{b}$ 로 다시 쓰면, 안정성을 결정하는 것은 행렬 $\mathbf{A}$ 다. $\mathbf{A}$ 가 **양의 정부호(positive definite)** 이면 ($\mathbf{y}^\top \mathbf{A}\mathbf{y} > 0$ for all $\mathbf{y} \ne 0$) 안정성이 보장되고 $\mathbf{A}^{-1}$ 도 존재한다. on-policy 분포 하에서 $\mathbf{A} = \mathbf{X}^\top \mathbf{D}(\mathbf{I} - \gamma\mathbf{P})\mathbf{X}$ 의 핵심 행렬 $\mathbf{D}(\mathbf{I} - \gamma\mathbf{P})$ 가 양의 정부호임이 증명된다.

**TD 고정점의 오차 한계 (연속 과제):**

$$
\overline{VE}(\mathbf{w}_{TD}) \le \frac{1}{1-\gamma}\,\min_{\mathbf{w}} \overline{VE}(\mathbf{w})
$$

즉 TD의 점근 오차는 MC가 달성하는 최소 오차의 $\frac{1}{1-\gamma}$ 배 이내다. $\gamma$ 가 1에 가까우면 이 확장 계수가 커져 점근 성능 손실이 클 수 있다. 반면 TD는 분산이 훨씬 작아 더 빠르다.

> **중요(on-policy 조건):** 이 수렴 결과들은 상태가 **on-policy 분포에 따라** 업데이트된다는 점에 결정적으로 의존한다. 다른 분포를 쓰면 함수 근사 부트스트래핑이 무한대로 **발산**할 수도 있다(Ch. 11). 이것이 이 코스가 "on-policy" prediction에 한정되는 이유다.

---

## 모듈 종합 — 흐름 한 줄 요약

표 → 파라미터화된 함수($\hat{v}(s,\mathbf{w})$) → 목적함수 $\overline{VE}$ 정의 → SGD로 최소화 → MC는 진짜 경사·불편 목표(전역 최적) / TD는 준-경사·편향 목표(TD 고정점, 빠르지만 점근 오차 존재) → 선형 근사에서 둘 다 수렴 보장(단 on-policy 분포 하에서).

다음 모듈(Module 3)에서는 선형 근사의 성패를 좌우하는 **특징 $\mathbf{x}(s)$ 를 어떻게 구성할 것인가**(coarse coding, tile coding)와, 특징을 자동으로 학습하는 **비선형 근사(신경망)** 로 확장한다.

---

## 개인 인사이트 / 캡스톤 연결

**1. Course 2의 tabular 한계가 이번 모듈에서야 정리되었다.**

Course 2 Phase 2 실험에서 8×8 그리드(64 상태)로 tabular Q-learning을 돌렸을 때는 표 하나로 충분했다. 당시에는 "상태가 많아지면 어떻게 하지"라는 의문을 잠깐 떠올리고 넘어갔는데, 이번 모듈에서 $\hat{v}(s, \mathbf{w})$ 라는 형태로 그 답이 명시적으로 정리되었다. 표가 단지 특징 벡터가 one-hot인 선형 근사의 특수 케이스라는 점(Exercise 9.1)을 알고 나니, 지금까지 배운 tabular 방법들이 별개의 기법이 아니라 같은 틀의 한쪽 끝이었다는 게 분명해졌다.

**2. 9·10주차 Cliff Walking 실험이 "TD 고정점 ≠ 전역 최적"으로 다시 설명된다.**

9·10주차에서 Q-learning이 이론적 최단 경로를 학습했지만 ε이 커질수록 실제 성능이 무너지는 것(ε=0.1에서 SARSA -26.11 vs Q-learning -56.29)을 "학습하는 것과 실행되는 것이 다르면 성능 보장이 안 된다"로 정리했었다. 이번 모듈의 TD 고정점 $\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}$ 와 오차 한계 $\overline{VE}(\mathbf{w}_{TD}) \le \frac{1}{1-\gamma}\min_\mathbf{w}\overline{VE}(\mathbf{w})$ 는, 함수 근사에서도 TD가 수렴하는 지점이 전역 최적과 다르다는 같은 결을 수식으로 보여준다. tabular Cliff Walking에서 직관으로 본 트레이드오프가 함수 근사 차원에서 정량적 한계로 다시 나타난 셈이다.

**3. 캡스톤 navigation의 연속 상태 공간에서는 표가 원천적으로 불가능하다.**

캡스톤으로 잡은 Isaac Lab navigation은 로봇 위치·속도·센서 관측이 모두 연속값이다. 교재가 든 "카메라 이미지의 경우의 수가 우주의 원자 수보다 많다"는 비유가 정확히 이 상황이고, tabular로는 접근 자체가 불가능하다. 이번 모듈은 캡스톤이 왜 함수 근사를 전제로 출발해야 하는지를 분명히 해준다 — 선택의 문제가 아니라 전제 조건이다.

**4. $\overline{VE}$ 의 $\mu$ 가중은 중간보고서의 "보상 = 목적함수" 원칙과 닮았다.**

$\overline{VE}(\mathbf{w}) = \sum_s \mu(s)[v_\pi(s) - \hat{v}(s,\mathbf{w})]^2$ 에서 $\mu(s)$ 가 "어떤 상태의 오차를 더 중요하게 볼지"를 정의하고, 그 선택이 최종 해의 형태를 바꾼다(Example 9.1의 편향된 그룹 추정값). 이는 중간보고서에서 도출한 MDP 설계 원칙 2번 "Reward는 목적함수와 직접 대응"과 구조적으로 같다. 목적함수에 무엇을 얼마나 반영하느냐가 학습 결과를 좌우한다는 점에서, 보상 설계와 $\overline{VE}$ 의 상태 가중은 같은 사고방식을 요구한다. 캡스톤 보상 설계 단계에서 이 관점을 그대로 가져갈 수 있을 것 같다.


