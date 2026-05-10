# 실험 D_failure: 빠르게 가

## 입력
- 자연어 명령: "빠르게 가"
- 가설: 목표 미명시 때문에 LLM이 `goal=null`을 반환하고 builder에서 controlled failure가 발생한다.
- 성공 기준: 실패 단계가 parse 또는 build로 명확하게 기록된다.

## LLM 파싱 결과
```json
{
  "grid_size": [8, 8],
  "start": [0, 0],
  "goal": null,
  "obstacles": [],
  "soft_avoid": [],
  "preference": "shortest",
  "interpretation_notes": "빠르게는 preference=shortest로 해석했지만 목표가 명시되지 않아 goal=null"
}
```

해석 노트: LLM은 목적지를 임의로 만들지 않고 `goal=null`로 보존했다. "빠르게"는 `preference=shortest`로 해석됐지만, 목표가 없으므로 MDP를 만들 수 없다.

## 학습 결과
- 상태: `build_failed`
- 실패 원인: `Goal is underspecified`
- 학습 미실행

## 가설 vs 실제 결과
가설은 충족됐다. 실패는 LLM JSON 파싱 단계가 아니라 builder 단계에서 발생했고, 원인은 명확히 "목표 미명시"로 기록됐다. 이는 사용자가 재질문을 받아야 하는 케이스를 시스템이 식별할 수 있음을 보여준다.

## 발견과 한계
이 실패는 나쁜 결과가 아니라 중요한 정찰 결과다. 목표가 없는 명령에 대해 임의 목적지를 hallucinate하지 않고 멈추는 것이 바람직하다. 다음 버전에서는 `goal=null`일 때 "어디로 갈까요?" 같은 clarification loop를 붙이는 것이 자연스럽다.
