---
title: "M3 실습 - 신경망 vs 타일코딩"
course: "코스3 - 함수근사 예측·제어"
module: "Module 3 실습"
type: 실습
tags:
  - rl/코스3-함수근사
  - 유형/실습
  - 개념/신경망
  - 개념/tile-coding
  - 개념/Adam
  - 개념/sample-efficiency
---

# 신경망 기반 Semi-gradient TD의 가치함수 근사 성능 분석 (Neural Network vs Tile-coding)

## 1. Introduction and Background

본 실험은 이전 과제와 동일한 500-State Random Walk 환경에서 고정된 정책의 상태 가치함수(state-value function)를 근사하는 policy evaluation(prediction) 문제를 다룬다. 다만 이전 과제가 state aggregation이라는 수작업 feature를 사용했다면, 본 과제에서는 함수근사기로 **신경망(neural network)** 을 사용하고, 같은 문제에서 두 방식의 학습 효율을 비교한다. 목표는 무작위 정책을 따를 때 각 상태가 장기적으로 가지는 기대 보상, 즉 $v_\pi(s)$를 추정하는 것이다.

환경은 이전과 같다. 비종단 상태는 1번부터 500번까지 존재하고, agent는 중앙인 state 250에서 시작하며, 매 step마다 좌우를 같은 확률(0.5)로 선택한다. 왼쪽 terminal에 도달하면 보상 −1을, 오른쪽 terminal에 도달하면 보상 +1을 받는다. 따라서 참 가치함수는 상태 번호가 커질수록 −1에서 +1로 부드럽게 증가하는, 거의 직선에 가까운 형태를 띤다. 이 "인접한 상태는 가치가 비슷하다"는 1차원의 매끄러운 구조가 본 실험을 이해하는 핵심 배경이다.

본 실험에서 사용한 함수근사기는 은닉층이 하나인 신경망이다. 입력은 상태 번호 그 자체가 아니라 상태의 **one-hot encoding**(길이 500)이다. 상태 번호를 정수로 직접 넣으면 "250과 251은 가깝다"는 사전 지식이 모델에 주입되는데, 본 과제는 의도적으로 이 정보를 제거하기 위해 one-hot을 사용한다. 즉 신경망은 인접 상태가 비슷하다는 사실조차 데이터로부터 스스로 배워야 한다. 은닉층은 100개의 ReLU 유닛으로 구성되며, 출력은 은닉 유닛의 선형 결합으로 계산되는 단일 가치 추정값이다. 입력 $s$에 대한 가치 $v$는 다음과 같이 계산된다.

$$\psi = sW^{[0]} + b^{[0]}, \qquad x = \max(0, \psi), \qquad v = xW^{[1]} + b^{[1]}$$

학습 알고리즘은 이전과 같은 Semi-gradient TD(0)이다. 한 step 이후의 보상과 다음 상태의 가치 추정치로 현재 상태의 추정치를 갱신하며, TD error와 weight update는 다음과 같다.

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \, \delta_t \, \nabla \hat{v}(S_t, \mathbf{w}_t)$$

이전 과제에서는 feature가 one-hot이라 $\nabla \hat{v}$가 단순했지만, 신경망에서는 가치 추정값 $\hat{v}$가 여러 층의 weight에 의존하므로 $\nabla \hat{v}$를 backpropagation으로 계산해야 한다. 또한 'semi-gradient'라는 이름이 붙는 이유도 동일하다. TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$ 안의 $\hat{v}(S_{t+1}, \mathbf{w}_t)$도 현재 weight에 의존하지만, 업데이트 시점에는 이를 고정된 목표값처럼 취급하고 그 부분의 gradient는 무시하기 때문이다.

---

## 2. Method and Experimental Setup

구현은 신경망의 순전파·역전파·가중치 갱신을 담당하는 보조 함수들로 구성된다. `get_value()`는 위의 순전파 식에 따라 one-hot 입력으로부터 가치 $v$를 계산한다. 입력이 희소(one-hot)하다는 점을 이용해 행렬 곱을 빠르게 처리하는 보조 함수 `my_matmul()`을 사용하였다. `get_gradient()`는 backpropagation으로 $v$를 각 weight로 미분한 gradient를 반환하며, ReLU의 미분은 $x>0$인 위치에서 1, 그 외에서 0인 indicator로 처리하였다.

가중치 갱신은 기본적으로 stochastic gradient descent(SGD) 형태이지만($\mathbf{w} \leftarrow \mathbf{w} + \alpha \, g_t$, 여기서 $g_t = \delta_t \nabla \hat{v}$), 실제 실험에서는 더 효율적인 **Adam** 알고리즘을 사용하였다. Adam은 업데이트의 1차·2차 모멘트 추정치 $\mathbf{m}, \mathbf{v}$를 유지하여 모멘텀과 파라미터별 적응적 step-size를 함께 적용한다.

$$\mathbf{m}_t = \beta_m \mathbf{m}_{t-1} + (1 - \beta_m) g_t, \qquad \mathbf{v}_t = \beta_v \mathbf{v}_{t-1} + (1 - \beta_v) g_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_m^t}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_v^t}, \qquad \mathbf{w}_t = \mathbf{w}_{t-1} + \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\,\hat{\mathbf{m}}_t$$

신경망의 가중치는 평균 0, 표준편차 $\sqrt{2 / (\text{입력 노드 수})}$인 정규분포로 초기화하였다. 이는 ReLU를 사용할 때 흔히 쓰는 초기화 방식으로, 층을 지날수록 출력이 지나치게 커지거나 작아지는 것을 막아준다.

평가 지표는 이전 과제와 동일하게 RMSVE(Root Mean Squared Value Error)를 사용하였다. 학습된 가치함수 $\hat{v}(s, \mathbf{w})$와 참 가치함수 $v_\pi(s)$의 차이를, 정책을 따를 때의 상태 방문 분포 $\mu(s)$로 가중하여 측정한다. 값이 작을수록 true value에 가깝다는 의미이다.

$$\text{RMSVE} = \sqrt{\sum_{s \in \mathcal{S}} \mu(s)\,[v_\pi(s) - \hat{v}(s, \mathbf{w})]^2}$$

실험 설정은 다음과 같다.

| 항목                   | 설정                                                       |
| -------------------- | -------------------------------------------------------- |
| Environment          | 500-State Random Walk                                    |
| Learning method      | Semi-gradient TD(0)                                       |
| Value representation | Neural Network (은닉층 1개, ReLU 100 units, 선형 출력)        |
| Network input        | state의 one-hot encoding (길이 500)                         |
| Optimizer            | Adam (step-size 0.001, β_m 0.9, β_v 0.999, ε 1e-4)        |
| Discount factor      | 1.0                                                      |
| Episodes per run     | 1000 (직접 실행) / 5000 (과제 제공)                            |
| Number of runs       | 20                                                       |
| Evaluation metric    | RMSVE                                                    |
| 비교 대상 (Comparison)  | Semi-gradient TD with Tile-coding (50 tilings × 6 tiles, step-size 0.1/50) |

본 실험에서는 신경망 TD를 20개의 run에 대해 평균하여 학습된 approximate value와 RMSVE의 변화를 관찰하였다. 1000 episodes 결과는 직접 실행하였고, 학습이 충분히 진행되었을 때의 성능을 보기 위해 5000 episodes(20 runs) 결과를 함께 비교하였다. 마지막으로 동일한 문제에서 tile-coding 기반 semi-gradient TD와 성능을 비교하였다.

---

## 3. Results and Discussion

<!-- ============================================================ -->
![[c3m3lab-image.png]]
**Figure 1.** 100개 hidden unit을 가진 신경망 TD의 학습 결과 (20 runs 평균). 위쪽은 1000 episodes, 아래쪽은 과제에서 제공한 5000 episodes 결과이다. 왼쪽 그래프는 학습된 approximate value와 true value($v_\pi$)를 비교한 것이고, 오른쪽 그래프는 episode 증가에 따른 RMSVE 변화를 나타낸다.

먼저 1000 episodes 결과를 보면, 학습된 가치함수는 전체적으로 상태 번호가 커질수록 증가하는 경향을 보인다. 즉 왼쪽 상태는 낮은 가치, 오른쪽 상태는 높은 가치라는 방향성은 제대로 학습하였다. 그러나 추정선이 참 가치함수에 딱 붙지는 않는다. 특히 양 끝에서 참값보다 덜 극단적으로 예측하여(상태 1 근처는 충분히 낮지 않고, 상태 500 근처는 충분히 높지 않다), 전체 기울기는 배웠지만 정확한 크기와 매끄러운 형태까지는 학습하지 못한 상태이다. 추정선이 상태마다 들쭉날쭉하게 떨리는 것도 아직 수렴하지 않았음을 보여준다. RMSVE 학습 곡선에서도 오차가 약 0.42에서 0.20 부근까지 꾸준히 내려가지만, 1000 episodes 시점에서도 여전히 감소 중이어서 학습이 끝나지 않았음을 알 수 있다.

5000 episodes 결과에서는 추정선이 1000 episodes일 때보다 참 가치함수에 더 가까워진다. RMSVE 역시 초반에 빠르게 내려간 뒤 천천히 감소하여 약 0.13~0.15 부근에서 진동한다. 즉 충분히 오래 학습하면 신경망 TD도 쓸 만한 근사를 얻기는 한다. 그러나 여전히 완전히 매끄러운 직선은 아니며 RMSVE도 0에 가깝지 않다. 정리하면, 신경망 TD는 학습 자체는 성공하지만(방향과 추세를 배우고 RMSVE가 감소한다), 이 단순한 random walk 문제에서는 수렴이 생각보다 느리고 결과도 다소 거칠다.

이 지점에서 자연스러운 질문은 "왜 신경망 TD가 느린가"이다. 이전 과제의 10-state aggregation이 약 100 episodes 안에 수렴했던 것과 대조적이다. 가장 큰 이유는 신경망이 처음부터 좋은 상태 표현을 갖고 있지 않다는 점이다. 입력이 one-hot이므로 신경망은 상태 1, 2, 3이 서로 가깝다는 사실을 모른 채 시작하며, 이 부드러운 1차원 구조를 수천 번의 TD 업데이트를 통해 스스로 배워야 한다. 반면 state aggregation은 애초에 인접한 상태들을 한 group으로 묶어 표현하므로, 한 상태를 업데이트하면 같은 group의 여러 상태가 함께 학습되는 강한 일반화 효과가 처음부터 내장되어 있다.

또한 신경망은 파라미터 수가 훨씬 많다. 입력 500개, hidden unit 100개이면 첫 번째 층의 weight만 해도 $500 \times 100 = 50{,}000$개이며, 여기에 bias와 출력층 weight가 더해진다. 이 많은 파라미터를 TD error라는 noisy한 학습 신호로 조금씩 조정해야 하므로 느릴 수밖에 없다. 게다가 TD target은 supervised learning처럼 고정된 정답이 아니라, 타깃 안의 $\hat{v}(S_{t+1}, \mathbf{w})$ 자체가 현재 신경망의 예측값인 '움직이는 목표'이다. 신경망처럼 복잡한 함수근사기를 쓰면 이런 bootstrapping이 학습을 더 느리고 불안정하게 만들 수 있고, 공유된 weight를 통해 상태들끼리 간섭이 생기기도 한다.

hidden unit 수를 바꾸면 어떻게 될지도 생각해 볼 수 있다. hidden unit을 줄이면 파라미터가 줄어 초기 학습이 빨라질 수 있고, 이 문제의 참 가치함수가 거의 직선에 가까운 단순한 형태이므로 굳이 큰 신경망이 필요하지 않을 수도 있다. 다만 너무 줄이면 모델의 표현력이 부족해져 참값을 제대로 근사하지 못하는 underfitting이 발생할 수 있다. 반대로 hidden unit을 늘리면 표현력(capacity)은 커지지만, 이 문제처럼 목표 함수가 단순한 경우에는 큰 이득이 없고 오히려 파라미터가 늘어 학습이 더 느려지고 noisy한 업데이트에 더 민감해질 수 있다. 즉 이 문제에서 "더 큰 신경망 = 더 빠른 학습"은 성립하지 않으며, 모델 크기는 무조건 클수록 좋은 것이 아니라 문제의 구조와 데이터 효율성에 맞아야 한다.

![[c3m3lab-image-1.png]]

**Figure 2.** Semi-gradient TD with Neural Network와 Semi-gradient TD with Tile-coding의 비교 (5000 episodes, 20 runs). 왼쪽은 학습된 state value, 오른쪽은 RMSVE 학습 곡선이다.

마지막으로 동일한 문제에서 tile-coding 기반 TD와 비교하면, tile-coding이 신경망보다 훨씬 빠르고 최종 오차도 더 낮다. 학습 곡선에서 tile-coding의 RMSVE는 초반에 급격히 떨어져 약 0.1 부근에 빠르게 도달하는 반면, 신경망은 천천히 내려가 5000 episodes 시점에도 tile-coding보다 높은 수준에 머문다. value plot에서도 tile-coding의 추정선은 참 가치함수에 거의 붙어 있는 반면, 신경망의 추정선은 전체 추세는 맞지만 훨씬 더 noisy하고 덜 정확하다.

그 이유는 tile-coding이 이 문제에 잘 맞는 feature를 처음부터 제공하기 때문이다. tile-coding은 상태공간을 여러 개의 겹치는 구획(tiling)으로 나누고 현재 상태가 어떤 tile들에 속하는지를 feature로 만든다(본 비교에서는 50개 tiling, 각 6개 tile 사용). 이 방식에서는 가까운 상태들이 비슷한 tile을 공유하므로, 한 상태를 업데이트하면 주변 상태의 추정값도 자연스럽게 함께 갱신된다. random walk의 참 가치함수가 상태 번호에 따라 부드럽게 증가하는 1차원 구조라는 점을 생각하면, 이 "인접 상태는 비슷하다"는 성질은 매우 강력한 inductive bias로 작동한다. 즉 tile-coding은 문제의 부드러운 구조를 처음부터 반영하는 반면, one-hot 입력의 신경망은 그 구조를 데이터로부터 학습해야 하므로 더 느리다. 결과적으로 5000 episodes 종료 시점에서 RMSVE가 더 낮은 방법은 tile-coding이다(대략 신경망 0.13~0.15, tile-coding 0.09~0.11 수준).

---

## 4. Conclusion

본 과제는 500-State Random Walk 환경에서 은닉층이 하나인 신경망(ReLU 100 units)과 Adam 알고리즘을 사용해 Semi-gradient TD로 상태 가치함수를 근사하고, 같은 문제에서 tile-coding 기반 TD와 학습 효율을 비교하였다. 신경망 TD는 학습 자체에는 성공하여, 가치함수의 전체적인 증가 패턴을 배우고 RMSVE를 꾸준히 낮추었다. 그러나 1000 episodes 시점에서는 아직 수렴하지 못해 추정값이 양 끝에서 참값보다 덜 극단적이고 전반적으로 noisy했으며, 5000 episodes까지 학습해도 RMSVE가 약 0.13~0.15 수준에 머물러 완전히 정확한 근사에는 이르지 못했다.

신경망 TD가 이전 과제의 10-state aggregation(약 100 episodes에 수렴)보다 훨씬 느린 이유는, 입력이 one-hot이라 인접 상태가 비슷하다는 구조를 처음부터 알지 못하고 데이터로부터 학습해야 하며, 파라미터 수가 많고(첫 층만 5만 개), TD target이 자기 자신의 예측에 의존하는 noisy한 '움직이는 목표'이기 때문이다. 또한 hidden unit 수를 무작정 늘린다고 빨라지는 것이 아니라, 이 문제처럼 목표 함수가 단순한 경우에는 과도하게 큰 모델이 오히려 학습을 더 느리고 불안정하게 만들 수 있다.

tile-coding과의 비교는 이 점을 분명히 보여준다. tile-coding은 겹치는 tiling을 통해 가까운 상태가 feature를 공유하도록 하여, random walk의 부드러운 1차원 구조에 맞는 inductive bias를 처음부터 제공한다. 그 결과 신경망보다 빠르게 학습할 뿐 아니라 5000 episodes 종료 시점의 최종 RMSVE도 더 낮았다.

결론적으로 본 실험의 의의는 "신경망이 나쁘다"가 아니라 **"신경망이 항상 최선의 선택은 아니다"** 라는 점에 있다. 신경망은 수작업 feature 없이도 표현을 스스로 만들어내는 강력한 함수근사기이며, 이미지·센서·연속 제어처럼 상태 표현이 복잡한 문제에서는 거의 필수적이다. 그러나 500-state random walk처럼 상태공간이 1차원이고 가치함수가 부드러운 단순한 문제에서는, tile-coding이나 state aggregation 같은 고전적 함수근사가 훨씬 sample efficient하다. 따라서 함수근사기는 무조건 강력한 것을 고르기보다, 문제의 구조와 주어진 sample 예산에 맞게 선택해야 한다.



---

## 🔗 관련 노트
- [[M3 - 특징 설계(Coarse·Tile·신경망)]]
- [[M2 실습 - 상태집계 랜덤워크 분석]]
