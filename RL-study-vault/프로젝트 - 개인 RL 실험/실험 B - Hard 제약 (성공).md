---
title: "실험 B - Hard 제약 (성공)"
course: "프로젝트 - 개인 RL 실험"
module: "nl-conditioned-grid 실험"
type: 실험
tags:
  - rl/프로젝트
  - 유형/실험
  - 개념/실험
  - 개념/hard제약
  - 개념/obstacles
  - 개념/성공사례
---

# 실험 B_constrained: 장애물 영역을 절대 피해서 (7,7)로 이동

## 입력
- 자연어 명령: "(7,7)로 가고 (3,3), (3,4), (4,3), (4,4) 영역은 절대 피해"
- 가설: 장애물을 우회하는 경로를 학습하고 obstacle 진입이 greedy rollout에서 0회가 된다.
- 성공 기준: greedy rollout 중 obstacle 진입 횟수 0회.

## LLM 파싱 결과
```json
{
  "grid_size": [8, 8],
  "start": [0, 0],
  "goal": [7, 7],
  "obstacles": [[3, 3], [3, 4], [4, 3], [4, 4]],
  "soft_avoid": [],
  "preference": "default",
  "interpretation_notes": "(3,3), (3,4), (4,3), (4,4) 영역을 obstacles로 분류"
}
```

해석 노트: "절대 피해"가 hard constraint로 해석되어 `obstacles`에 들어갔다. 이 매핑은 명세의 hard vs soft 구분과 일치한다.

## 학습 결과
- 최종 평가 평균 reward: `87.0`
- greedy rollout 평균 경로 길이: `14.0`
- 성공률: `1.0`
- obstacle 진입 횟수: `0`
- timeout 횟수: `0`
- 예시 경로: `(0,0) -> (0,1) -> (1,1) -> (1,2) -> (2,2) -> (3,2) -> (4,2) -> (5,2) -> (6,2) -> (6,3) -> (7,3) -> (7,4) -> (7,5) -> (7,6) -> (7,7)`

## 가설 vs 실제 결과
가설은 충족됐다. 정책은 `(3,3), (3,4), (4,3), (4,4)` 블록 왼쪽으로 우회했고, greedy rollout에서 obstacle 진입은 `0`회였다. 경로 길이도 `14.0`으로 최단 길이를 유지했는데, 이는 장애물 배치가 중앙을 막지만 가장자리 우회를 크게 강제하지 않는 형태였기 때문이다.

## 발견과 한계
hard constraint는 현재 파이프라인에서 안정적으로 작동한다. 다만 builder가 장애물을 transition reset으로 처리하므로, 실제 로봇/시뮬레이터로 확장할 때는 "진입 불가능한 벽"과 "충돌 후 reset"을 구분해야 한다.



## 📊 결과 이미지

![[projexpB-learning_curve.png]]

![[projexpB-policy.png]]


---

## 🔗 관련 노트
- [[실험 A - 명확한 목표 (성공)]]
- [[실험 C - 모호한 선호도 (부분성공)]]
- [[실험 D - 목표 미명시 (제어된 실패)]]
- [[실험 - 실패 분류·위험도 분석]]
