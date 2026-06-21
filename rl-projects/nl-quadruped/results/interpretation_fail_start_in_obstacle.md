# fail_start_in_obstacle

- 해석 노트: 시작 [4,4]가 금지 박스 안 — 출발부터 충돌 상태, 계획 단계에서 거부되어야 함
- 단계 결과: **planning_failed**
- 성공: False  (Start (4.0, 4.0) is inside an inflated forbidden region)
- 목표: [7.5, 7.5]  / 속도: normal / preference: default
- forbidden: 1개, soft_avoid: 0개
- waypoint 수: 0, 경로 길이: 0.0 m, forbidden 충돌: 0회