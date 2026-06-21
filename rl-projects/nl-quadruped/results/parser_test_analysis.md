# Parser Test Analysis

| id | command | expected | parse | goal | speed | pref | forb | soft | notes/error |
|---|---|---|---|---|---|---|---|---|---|
| T01 | 오른쪽 위 구석으로 가 | clear | ok | [7.5, 7.5] | normal | default | 0 | 0 | 오른쪽 위 구석을 [7.5,7.5]로 해석 |
| T02 | 오른쪽 위 구석으로 가되 중앙은 절대 피해서 천천히 | clear_hard_slow | ok | [7.5, 7.5] | slow | default | 1 | 0 | 오른쪽 위 구석을 [7.5,7.5]로 해석. 중앙을 [3,3]~[5,5] 박스로 보고 '절대 피해'는 forbidden_region으로 처리, clearance=0.3 적용. '천천히'는 speed=slow |
| T03 | (6,6)까지 안전하게 가, (3,3) 근처는 위험해 | soft_safe | ok | [6.0, 6.0] | normal | safe | 0 | 1 | '안전하게'는 preference=safe, '(3,3) 근처 위험'은 반경1 원형 soft_avoid |
| T04 | 빠르게 가 | underspecified | ok | None | fast | shortest | 0 | 0 | '빠르게'는 speed=fast, preference=shortest로 해석. 목표 위치가 명시되지 않아 goal=null로 설정 |
| T05 | 왼쪽 위 구석으로 최대한 빨리 | clear_fast | ok | [0.5, 7.5] | fast | shortest | 0 | 0 | '왼쪽 위 구석'을 [0.5,7.5]로 해석, '최대한 빨리'는 speed=fast 및 preference=shortest로 해석 |
| T06 | 가운데로 가 | ambiguous_center | ok | [4.0, 4.0] | normal | default | 0 | 0 | '가운데'를 중앙 [4.0,4.0]으로 해석 |
| T07 | 오른쪽 아래 구석까지 가는데 중앙 구역은 조심해서 돌아가 | soft_safe | ok | [7.5, 0.5] | normal | safe | 0 | 1 | '오른쪽 아래 구석'을 [7.5,0.5]로 해석. '중앙 구역 조심해서 돌아가'는 중앙을 [3,3]~[5,5] 박스로 보고 soft_avoid_regions에 추가, '조심해서'는 preference=safe로 해석 |
| T08 | (2,7)에서 출발해서 왼쪽 아래 구석까지 천천히 안전하게 | clear_start_slow_safe | ok | [0.5, 0.5] | slow | safe | 0 | 0 | 출발점을 [2.0,7.0]으로 해석, '왼쪽 아래 구석'을 [0.5,0.5]로 해석, '천천히'는 speed=slow, '안전하게'는 preference=safe |
| T09 | 장애물 없이 목표까지 | underspecified | ok | None | normal | default | 0 | 0 | '장애물 없이'는 장애물이 없는 상황을 의미하므로 forbidden_regions 없음. '목표까지'라고 했으나 구체적인 목표 위치가 명시되지 않아 goal=null로 설정. 목표 좌표를 추가로 지정해야 함. |
| T10 | 왼쪽 벽은 절대 붙지 말고 오른쪽 끝 가운데로 가 | clear_hard | ok | [7.5, 4.0] | normal | default | 1 | 0 | '오른쪽 끝 가운데'는 x=7.5(오른쪽 끝), y=4.0(세로 중앙)인 [7.5,4.0]으로 해석. '왼쪽 벽은 절대 붙지 말고'는 왼쪽 벽(x=0) 근처를 forbidden_region으로 처리하여 x=0~0.5 구간을 금지 박스로 설정. clearance=0.5로 벽 여유 확보. |