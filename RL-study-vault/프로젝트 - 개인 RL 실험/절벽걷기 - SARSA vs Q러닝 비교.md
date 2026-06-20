---
title: "절벽걷기 - SARSA vs Q러닝 비교"
course: "프로젝트 - 개인 RL 실험"
module: "cliff-walking"
type: 프로젝트
tags:
  - rl/프로젝트
  - 유형/프로젝트
  - 개념/SARSA
  - 개념/Q러닝
  - 개념/온폴리시
  - 개념/오프폴리시
  - 개념/절벽걷기
---

# Cliff Walking: SARSA vs Q-Learning

## 핵심 발견
1. epsilon=0.1에서 SARSA의 마지막 100 episode 평균 return은 -26.11입니다.
2. Q-learning은 off-policy Bellman backup을 사용하여 경험에서 학습하며, epsilon=0.1 평균 return은 -56.29입니다.
3. epsilon=0.3에서도 Q-learning 평균 return은 -227.39입니다.

## 왜 SARSA가 더 긴 경로를 선호하는가
SARSA는 on-policy 방법이라 epsilon-greedy로 실제 실행되는 탐험 행동의 비용까지 업데이트에 반영한다. 절벽 근처의 최단 경로는 한 번의 탐험 행동으로 큰 손실을 낼 수 있으므로, SARSA는 그 위험을 Q 값에 포함해 더 안전한 우회 경로를 선호하는 경향이 있다. 반면 Q-learning은 다음 상태에서 항상 greedy 행동을 한다고 가정하는 off-policy 업데이트라, 구현하면 절벽 바로 위의 더 짧은 경로를 더 쉽게 선택한다.

## 산출물
- `results/learning_curves.png`
- `results/policy_sarsa_eps01.png`
- `results/policy_qlearning_eps01.png`
- `results/summary.json`



## 📊 결과 이미지

![[projcliff-results-learning_curves.png]]

![[projcliff-results-policy_sarsa_eps01.png]]

![[projcliff-results-policy_qlearning_eps01.png]]


---

## 🔗 관련 노트
- [[M4 - TD 제어(SARSA·Q러닝)]]
- [[M4 실습 - Q러닝·Expected SARSA Cliff World]]
