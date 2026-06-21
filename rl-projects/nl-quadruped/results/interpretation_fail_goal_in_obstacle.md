# fail_goal_in_obstacle

- 해석 노트: 목표 [4,4]가 금지 박스 [3,3]~[5,5] 안에 있음 — 도달 불가, 계획 단계에서 거부되어야 함
- 단계 결과: **planning_failed**
- 성공: False  (Goal (4.0, 4.0) is inside an inflated forbidden region)
- 목표: [4.0, 4.0]  / 속도: normal / preference: default
- forbidden: 1개, soft_avoid: 0개
- waypoint 수: 0, 경로 길이: 0.0 m, forbidden 충돌: 0회