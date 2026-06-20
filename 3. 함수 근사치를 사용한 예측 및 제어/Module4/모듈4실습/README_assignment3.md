# Tile Coding 기반 Semi-gradient Sarsa의 Mountain Car 제어 성능 분석

## 1. Introduction and Background

본 실험은 앞선 두 과제와 달리 정책을 평가하는 prediction 문제가 아니라, 더 나은 행동 정책을 직접 학습하는 **control 문제**를 다룬다. 환경은 Mountain Car로, 힘이 약한 자동차가 언덕 꼭대기의 목표 지점에 도달하도록 만드는 과제이다. 자동차는 단순히 오른쪽으로만 가속해서는 언덕을 오르지 못하고, 왼쪽으로 갔다가 다시 오른쪽으로 밀면서 운동량(momentum)을 만들어야 목표에 도달할 수 있다. 따라서 agent는 위치와 속도에 따라 언제 가속 방향을 바꿔야 하는지를 스스로 학습해야 한다.

Mountain Car의 상태는 두 개의 연속값, 즉 위치(−1.2 ~ 0.5)와 속도(−0.07 ~ 0.07)로 주어진다. 위치와 속도가 실수값이므로 가능한 상태의 수는 사실상 무한하다. "위치 −0.5321, 속도 0.0123일 때는 오른쪽으로 가라"처럼 모든 경우를 표(table)로 외우는 것은 불가능하다. 그래서 연속 상태공간을 다룰 함수근사 방법이 필요하며, 본 과제에서는 **Tile Coding**을 사용한다. 보상은 매 step마다 −1이므로, 목표에 빨리 도달할수록(=불필요한 움직임을 줄일수록) 총 보상이 커진다. 즉 학습이 잘 될수록 한 episode를 끝내는 데 걸리는 step 수가 줄어든다.

Tile Coding은 연속 공간을 여러 개의 격자(tiling)로 나누고, 현재 상태가 각 tiling에서 어느 칸(tile)에 속하는지를 feature로 만드는 방식이다. 이때 두 개의 핵심 파라미터가 있다. `num_tiles`는 하나의 격자를 얼마나 잘게 나눌지(해상도)를 결정하고, `num_tilings`는 그런 격자판을 몇 겹으로 겹쳐 놓을지를 결정한다. 여러 tiling을 조금씩 어긋나게 겹쳐 놓기 때문에, 가까운 두 상태는 많은 tile을 공유하고 멀리 떨어진 상태는 적게 공유한다. 이 성질 덕분에 Tile Coding은 상태를 구별하는 능력(discrimination)과 비슷한 상태끼리 정보를 나누는 능력(generalization)을 동시에 제공한다.

학습 알고리즘은 on-policy control 방법인 Semi-gradient Sarsa이다. 각 행동가치 $\hat{q}(s,a)$는 해당 상태에서 활성화된 tile들의 weight 합으로 계산된다. action마다 별도의 weight vector를 두어, $\hat{q}(s,a) = \sum_{i \in \text{active tiles}(s)} w[a][i]$로 표현한다. Tile feature는 활성 tile에서 1, 그 외에서 0인 이진(binary) 벡터이므로 gradient가 활성 tile 위치에서 1이 되어, 업데이트가 활성 tile의 weight에만 적용된다.

---

## 2. Method and Experimental Setup

Tile Coding 구현에서는 먼저 위치와 속도를 각자의 범위에 맞춰 $[0, \text{num\_tiles}]$ 구간으로 정규화한 뒤, `tiles3` 라이브러리의 해시 테이블(IHT)을 통해 활성 tile들의 인덱스를 얻는다. action에 대한 처리는 weight vector의 크기를 (num_actions, iht_size)로 두는 단순한 방식을 사용하였다. 즉 action별로 하나의 weight vector를 두고, tile마다 하나의 weight를 둔다. Mountain Car의 action은 좌·중립·우 가속의 3가지이다.

행동 선택은 $\varepsilon$-greedy로, 대부분은 행동가치가 가장 큰 action을 선택하되(동점은 무작위로 깸) 일정 확률로 무작위 탐험을 한다. Sarsa는 State–Action–Reward–State–Action의 순서대로 다음에 실제로 선택한 행동 $A'$의 가치를 사용해 TD error를 계산한다.

$$\delta_t = R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}) - \hat{q}(S_t, A_t)$$

$$w[A_t][\text{active tiles}(S_t)] \leftarrow w[A_t][\text{active tiles}(S_t)] + \alpha \, \delta_t$$

목표 상태에 도달한 terminal step에서는 다음 상태의 가치가 없으므로 $\delta_t = R_{t+1} - \hat{q}(S_t, A_t)$로 갱신한다. step-size는 $\alpha = 0.5 / \text{num\_tilings}$로 설정하여, tiling 수가 달라져도 한 번의 업데이트로 가치가 바뀌는 총량이 비슷하게 유지되도록 하였다. 할인율은 $\gamma = 1.0$이다.

실험은 두 단계로 진행하였다. 먼저 기본 설정(8 tilings, 8×8 tiles)으로 Sarsa agent가 학습하는지 확인하였고, 다음으로 tile coding 구성만 다르게 한 세 가지 설정을 비교하였다. 세 설정은 모두 **동일하게 512개의 feature**를 만들지만, 그 feature를 상태공간에 분배하는 방식이 다르다는 점이 핵심이다.

| 항목                 | 설정                                                          |
| ------------------ | ----------------------------------------------------------- |
| Environment        | Mountain Car (위치 −1.2~0.5, 속도 −0.07~0.07, 연속)             |
| Problem type       | Control (정책 학습)                                            |
| Learning method    | Semi-gradient Sarsa (on-policy)                             |
| Value representation | Tile Coding 기반 Linear Function Approximation              |
| Actions            | 3 (좌 / 중립 / 우 가속)                                        |
| Reward             | 매 step −1                                                  |
| Step-size α        | 0.5 / num_tilings                                          |
| Discount factor γ  | 1.0                                                        |
| 기본 실험            | 8 tilings × 8×8 tiles, 10 runs × 50 episodes               |
| 비교 실험            | (16 tiles × 2 tilings), (4 tiles × 32 tilings), (8 tiles × 8 tilings), 각각 512 features, 20 runs × 100 episodes |
| Evaluation metric  | episode당 step 수 (낮을수록 좋음)                              |

---

## 3. Results and Discussion

<!-- ============================================================ -->
<!-- ▼▼▼ 이미지 삽입 위치 ① : 기본 설정(8 tilings × 8×8) 학습 곡선 ▼▼▼ -->
<!--                                                            -->
<!--   여기에 기본 Sarsa agent의 학습 곡선 이미지를 넣으세요.    -->
<!--   (x축: episode / y축: steps per episode)                  -->
<!--                                                            -->

[ 📷 이미지 ① — 기본 설정(8 tilings × 8×8 tiles) 학습 곡선 (여기에 삽입) ]

<!-- ▲▲▲ 이미지 삽입 위치 ① 끝 ▲▲▲                              -->
<!-- ============================================================ -->

**Figure 1.** 8 tilings × 8×8 tiles 설정에서의 학습 곡선 (10 runs 평균). x축은 episode, y축은 한 episode를 끝내는 데 걸린 step 수이다.

먼저 기본 설정의 학습 곡선을 보면, episode가 반복될수록 한 episode를 끝내는 데 걸리는 step 수가 처음 약 1400 step 부근에서 점차 줄어들어 약 200 step 수준으로 수렴한다. 즉 처음에는 목표까지 가는 데 오래 걸렸지만, 학습이 진행되면서 더 짧은 시간 안에 목표에 도달하게 되었다. 보상이 매 step −1이라는 점을 생각하면, 이는 agent가 큰 보상을 직접 받는 법을 배운다기보다 불필요한 움직임을 줄이는 방향으로 정책을 개선했다고 해석할 수 있다. 곡선이 일관되게 감소한다는 사실 자체가, Sarsa agent가 위치와 속도에 따라 적절한 행동가치를 학습하고 있음을 보여준다.

<!-- ============================================================ -->
<!-- ▼▼▼ 이미지 삽입 위치 ② : 세 가지 Tile Coding 구성 비교 ▼▼▼ -->
<!--                                                            -->
<!--   여기에 세 구성 비교 그래프를 넣으세요.                    -->
<!--   파랑: num_tiles=16, num_tilings=2                        -->
<!--   주황: num_tiles=4,  num_tilings=32                       -->
<!--   초록: num_tiles=8,  num_tilings=8                        -->
<!--                                                            -->

[ 📷 이미지 ② — 세 가지 Tile Coding 구성 비교 (여기에 삽입) ]

<!-- ▲▲▲ 이미지 삽입 위치 ② 끝 ▲▲▲                              -->
<!-- ============================================================ -->

**Figure 2.** Tile Coding 구성에 따른 학습 곡선 비교 (20 runs 평균). 파란색은 num_tiles=16·num_tilings=2, 주황색은 num_tiles=4·num_tilings=32, 초록색은 num_tiles=8·num_tilings=8이다. 세 설정 모두 feature 수는 512로 같다.

다음으로 세 가지 tile coding 구성을 비교한 결과, 세 곡선 모두 시간이 지나면서 step 수가 감소하여 Sarsa가 어떤 구성에서도 학습은 한다는 점을 확인하였다. 그러나 학습 속도와 최종 성능에는 뚜렷한 차이가 나타났다. 가장 빠르고 안정적으로 좋아진 것은 주황색(num_tiles=4, num_tilings=32)이고, 초록색(8×8)이 그 중간이며, 가장 늦게 배우고 최종 성능도 가장 나빴던 것은 의외로 가장 세밀한 파란색(num_tiles=16, num_tilings=2)이다.

이 결과가 본 실험의 핵심 직관을 보여준다. 상태를 세밀하게 쪼갠다고 해서 무조건 좋은 것이 아니다. num_tiles=16처럼 격자를 너무 잘게 나누면 각 칸이 지나치게 독립적이 되어, 한 상태에서 배운 지식이 인접한 상태로 잘 퍼지지 않는다. 예컨대 위치 −0.50에서 배운 행동은 위치 −0.51에서도 비슷하게 쓸 수 있어야 하는데, 너무 촘촘하게 나누면 모델이 이를 "서로 다른 칸"으로 보아 따로 학습하게 되고, 그만큼 학습이 느려진다. 반면 num_tiles=4, num_tilings=32 설정은 격자 하나하나는 거칠지만 여러 겹의 tiling으로 상태를 표현하기 때문에, 가까운 상태끼리 tile을 많이 공유하여 정보가 자연스럽게 퍼진다. 그 결과 "골짜기 근처에서는 왼쪽으로 갔다가 다시 오른쪽으로 밀어야 한다" 같은, 넓은 구간에 걸친 행동 패턴을 더 빨리 익힌다. 세 설정의 feature 수가 모두 512로 같다는 점은 이 해석을 뒷받침한다. 즉 성능 차이는 feature가 많고 적음의 문제가 아니라, 같은 수의 feature를 상태공간에 어떻게 분배했는가의 문제이다.

이 실험에서 확인할 수 있는 교훈은 세 가지로 정리된다. 첫째, Sarsa agent가 실제로 학습하고 있다. 세 곡선 모두 episode가 반복될수록 step 수가 줄어든다. 둘째, 같은 Sarsa 알고리즘을 쓰더라도 tile coding 구성에 따라 결과가 크게 달라진다. 이는 머신러닝에서 모델 자체만큼이나 입력을 어떤 형태로 바꾸는가, 즉 feature representation이 중요하다는 점을 보여준다. 강화학습에서도 알고리즘 이름보다 "상태를 어떻게 보게 만들었는가"가 성능을 크게 좌우할 수 있다. 셋째, 일반화(generalization)가 중요하다. Mountain Car는 특정 좌표 하나만 정확히 맞춘다고 풀리는 문제가 아니라, 비슷한 위치·속도에서 비슷한 전략이 필요한 문제이다. 따라서 상태를 지나치게 세밀하게 나누기보다, 비슷한 상태끼리 묶어 정보를 공유하게 하는 표현이 더 효과적이다.

---

## 4. Conclusion

본 실험을 통해 연속 상태공간을 갖는 Mountain Car 문제에서 Tile Coding 기반 Sarsa agent가 episode가 반복될수록 목표 지점까지 도달하는 step 수를 줄여 나가는 것을 확인하였다. 이는 agent가 단순히 무작위로 행동하는 것이 아니라, 위치와 속도에 따라 적절한 행동가치를 학습하고 있음을 의미한다. 또한 동일한 Sarsa 알고리즘을 사용하더라도 tile coding의 구성 방식에 따라 학습 속도와 최종 성능이 크게 달라졌다. 특히 적은 수의 세밀한 tiling(16×16, 2 tilings)보다, 많은 수의 비교적 거친 tiling(4×4, 32 tilings)을 사용하는 설정이 더 빠르고 안정적인 성능을 보였다. 세 설정의 feature 수가 모두 512로 같았다는 점을 고려하면, 이는 연속 상태 문제에서 지나치게 세밀한 구분보다 인접 상태 간 정보를 공유할 수 있는 일반화 능력이 중요하다는 점을 보여준다.

이 결과는 연속적인 센서값을 다루는 강화학습 문제의 기본 설계 기준으로 활용할 수 있다. 로봇의 관절 각도·속도·위치, 또는 온도·압력·공정 변수처럼 대부분의 현실 신호는 연속값이라 그대로 Q-table에 넣을 수 없고 적절한 feature로 바꿔야 한다. Tile Coding은 딥러닝을 쓰기 전에 사용할 수 있는, 비교적 단순하고 해석 가능한 방법이다. 특히 "상태를 너무 세밀하게 쪼개면 오히려 학습이 느려질 수 있다", "비슷한 상태끼리 정보를 공유하게 만드는 표현이 중요하다"는 교훈은 로봇 제어, 공정 제어, 시뮬레이션 기반 학습에도 그대로 적용된다.

한 줄로 요약하면, 이 실험은 자동차가 언덕 오르는 법을 배우는 과정을 통해 강화학습에서 중요한 것이 알고리즘만이 아니라 상태를 어떻게 표현하느냐라는 점을 보여준 실험이라고 할 수 있다.
