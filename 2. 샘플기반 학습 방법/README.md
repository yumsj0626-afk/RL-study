# 2. 샘플 기반 학습 방법 (Sample-based Learning Methods)

> Coursera · University of Alberta · Reinforcement Learning Specialization **Course 2 / 4**

---

## 이 코스 한 줄 정의

Course 1에서는 환경 동역학 $p(s', r \mid s, a)$ 를 **안다고 가정**하고 벨만 방정식을 직접 풀었다.

Course 2는 그 가정을 버린다. 환경을 모를 때, **샘플링된 경험(experience)** 만으로 가치 함수와 정책을 학습하는 방법을 다룬다.

> Course 1 한계와의 연결: 상태 수가 많아지면 연립방정식 자체가 성립 불가 → 모델 없이 경험으로 추정하는 방법이 필요해진다. 이 코스가 그 답이다.

---

## 모듈 한눈에 보기

| Module | 주제 | 한 줄 요지 | 상태 |
| :--- | :--- | :--- | :---: |
| 1 | Welcome to the Course! | 코스 오버뷰 및 학습 목표 | ⬜ |
| 2 | Monte Carlo Methods for Prediction & Control | 에피소드를 끝까지 진행한 뒤, 실제 리턴으로 가치 추정 | ⬜ |
| 3 | TD Learning Methods for Prediction | 한 스텝 부트스트랩으로 가치 예측 (TD(0)) | ⬜ |
| 4 | TD Learning Methods for Control | SARSA / Q-learning, on-policy vs off-policy | ⬜ |
| 5 | Planning, Learning & Acting | Dyna — 모델 학습과 경험 학습의 결합 | ⬜ |

---

## 이 코스를 관통하는 두 축

모듈별 내용은 결국 아래 두 가지 비교로 정리된다. 노트를 채워가며 각 모듈이 이 축의 어디에 위치하는지 표시할 예정.

- **MC vs TD** — 에피소드 종료까지 기다려 실제 리턴으로 학습할 것인가, 한 스텝 부트스트랩으로 학습할 것인가
- **on-policy vs off-policy** — 행동 정책과 학습 대상 정책이 같은가, 다른가 (SARSA ↔ Q-learning)

---

## 디렉토리 구조

```
2. 샘플 기반 학습 방법/
├── README.md          ← (현재 파일) 코스 진입점
├── Module2/
│   └── Module2.md
├── Module3/
│   └── Module3.md
├── Module4/
│   └── Module4.md
└── Module5/
    └── Module5.md
```

---
