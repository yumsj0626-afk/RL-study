# NL-Conditioned Q-Learning: 구현 명세서

> **이 문서의 정체성**: 이 명세서는 Claude Code(또는 동등한 코드 에이전트)에게 그대로 전달하여 end-to-end 구현을 받기 위한 문서다. 각 단계는 (1) 무엇을 만들지, (2) 어떻게 만들지, (3) 결과를 어떻게 해석할지를 모두 포함한다.

> **프로젝트 정체성**: 이 프로젝트는 캡스톤이 아니다. 캡스톤 주제 결정을 위한 **정찰(reconnaissance)** 작업이다. 목표는 "자연어 → MDP 변환 파이프라인이 어디서 작동하고 어디서 깨지는가"를 가장 작은 환경에서 매핑하는 것이다.

---

## 0. 시스템 개요

### 아키텍처
```
[자연어 명령]
    │
    │  e.g. "왼쪽 위 모서리로 가되 가운데는 피해"
    ▼
┌─────────────────┐
│  LLM Parser     │  ← Anthropic Claude API (temperature=0)
│  (nl_parser.py) │     역할: 자연어 → 구조화 명세 JSON
└─────────────────┘
    │
    │  {"goal": [0,0], "obstacles": [...], "soft_avoid": [...]}
    ▼
┌─────────────────┐
│  Env Builder    │  ← JSON → NavGridEnv 인스턴스
│ (env_builder.py)│
└─────────────────┘
    │
    │  NavGridEnv(goal, obstacles, soft_avoid)
    ▼
┌─────────────────┐
│  Q-Learning     │  ← tabular Q-learning, ε-greedy
│   (agent.py)    │
└─────────────────┘
    │
    │  학습된 Q-table → greedy policy
    ▼
[정책 시각화 + 해석]
```

### 핵심 원칙
1. **LLM은 파서다, 정책이 아니다.** LLM은 자연어를 JSON으로 변환만 한다. 행동 결정은 Q-learning이 한다.
2. **재현성 필수.** 모든 실험에 random seed 고정, LLM 호출은 temperature=0.
3. **실패 사례가 자산이다.** LLM 파싱 실패, 학습 실패, 의도-환경 불일치 — 모두 기록한다.

---

## 1. 사전 준비

### 1.1 디렉토리 구조 (최종 형태)
```
RL-study/projects/
├── cliff-walking/                  # Phase 1 (압축)
│   ├── env.py
│   ├── agents.py
│   ├── run_experiments.py
│   ├── results/
│   └── README.md
│
└── nl-conditioned-grid/            # Phase 2 (메인)
    ├── env.py
    ├── nl_parser.py
    ├── env_builder.py
    ├── agent.py
    ├── run_experiments.py
    ├── prompts/
    │   └── parser_prompt_v1.txt
    ├── schemas/
    │   └── command_schema_v1.json
    ├── test_cases/
    │   └── commands.json
    ├── results/
    │   ├── exp_A_clear/
    │   ├── exp_B_constrained/
    │   ├── exp_C_ambiguous/
    │   ├── exp_D_failure/
    │   └── failure_taxonomy.md
    ├── architecture.md
    └── README.md
```

### 1.2 의존성
```
# requirements.txt
openai>=1.0
numpy>=1.24
matplotlib>=3.7
jsonschema>=4.0
```

### 1.3 환경 변수
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 2. Phase 1: Cliff Walking (압축본, 약 1.5일)

> **목표**: TD Learning 체화 + 환경 클래스 패턴 확립. Phase 2의 환경 클래스가 이 구조를 재사용한다.

### 2.1 `env.py` — Cliff Walking 환경

**구체 명세**
- 4×12 그리드, 시작 (3,0), 목표 (3,11), 절벽 (3,1)~(3,10)
- 행동: 0=상, 1=우, 2=하, 3=좌
- 보상: 절벽 진입 -100 + 시작점으로 reset, 목표 도달 0 + 종료, 그 외 -1
- **중요**: 생성자에서 `goal`, `cliff_cells`을 외부 주입 가능하게 (Phase 2 확장 대비)

```python
class CliffWalkingEnv:
    def __init__(self, height=4, width=12,
                 start=(3, 0), goal=(3, 11),
                 cliff_cells=None):
        # cliff_cells가 None이면 default (3,1)~(3,10)
        ...

    def reset(self) -> tuple:
        """returns initial state (row, col)"""

    def step(self, action: int) -> tuple:
        """returns (next_state, reward, done, info)"""

    def render_policy(self, Q, save_path=None):
        """matplotlib quiver로 greedy policy 시각화"""
```

### 2.2 `agents.py` — SARSA, Q-Learning

**핵심 업데이트 로직** (사용자가 직접 검증할 부분)
```python
# SARSA (on-policy)
Q[s, a] += alpha * (r + gamma * Q[s', a'] - Q[s, a])

# Q-learning (off-policy)
Q[s, a] += alpha * (r + gamma * max(Q[s', :]) - Q[s, a])
```

**클래스 구조**
```python
class TabularAgent:
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42):
        ...
    def act(self, state) -> int:
        """ε-greedy"""
    def update(self, ...):
        """SARSA or Q-learning - 서브클래스에서 구현"""

class SARSAAgent(TabularAgent): ...
class QLearningAgent(TabularAgent): ...
```

### 2.3 `run_experiments.py` — 6개 실험 자동화

**실행 매트릭스**
- 알고리즘: SARSA, Q-learning
- ε ∈ {0.01, 0.1, 0.3}
- 각 조합당 500 에피소드, seed=42

**산출물**
1. `results/learning_curves.png`: 6개 곡선 한 그래프
2. `results/policy_sarsa_eps01.png`, `policy_qlearning_eps01.png`: 화살표 정책
3. `results/summary.json`: 각 실험의 평균 누적 보상 마지막 100 에피소드

### 2.4 자동 생성될 README 템플릿

Claude Code는 실험 완료 후 다음 형식으로 README를 채운다:

```markdown
# Cliff Walking: SARSA vs Q-Learning

## 핵심 발견
1. ε=0.01에서 SARSA와 Q-learning의 차이는 [측정값]에 불과하다.
2. ε=0.3에서 Q-learning은 평균 [측정값]만큼 더 자주 절벽에 떨어진다.
3. SARSA는 항상 [몇] 행 위쪽 경로를 학습하고, Q-learning은 [몇] 행 경로를 학습한다.

## 왜 SARSA가 더 긴 경로를 선호하는가
[3-4줄로 자동 작성: SARSA가 평가하는 것이 ε-greedy 정책 자체이므로
탐험 중 절벽에 떨어지는 비용이 가치 함수에 반영된다는 점을 설명]
```

### 2.5 Phase 1 완료 체크리스트
- [ ] `env.py`: goal/cliff_cells 외부 주입 지원
- [ ] `agents.py`: SARSA, Q-learning 모두 동작
- [ ] 6개 실험 결과가 `results/`에 저장
- [ ] `learning_curves.png` 한 장에 6개 곡선
- [ ] README의 "왜 다른 경로를 학습하는가" 섹션 자동 작성

---

## 3. Phase 2: NL-Conditioned Q-Learning (메인, 약 4일)

### 3.1 Step 0: 명령 스키마 v1 정의

> **이 단계가 가장 중요하다.** 코딩 전에 스키마부터 확정한다. 스키마는 "자연어가 가진 정보 중 무엇을 MDP의 어디에 매핑할지"를 결정한다.

**파일**: `schemas/command_schema_v1.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Navigation Command Schema v1",
  "type": "object",
  "required": ["goal", "grid_size"],
  "properties": {
    "grid_size": {
      "type": "array",
      "items": {"type": "integer"},
      "minItems": 2,
      "maxItems": 2,
      "description": "[height, width]"
    },
    "start": {
      "type": "array",
      "items": {"type": "integer"},
      "minItems": 2,
      "maxItems": 2,
      "description": "시작 좌표 [row, col]. 미명시 시 [0, 0]"
    },
    "goal": {
      "type": "array",
      "items": {"type": "integer"},
      "minItems": 2,
      "maxItems": 2,
      "description": "목표 좌표 [row, col]"
    },
    "obstacles": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2
      },
      "description": "충돌 시 큰 페널티(-100) + reset되는 셀들"
    },
    "soft_avoid": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2
      },
      "description": "회피 권장 셀들 (페널티 -5, 통과는 가능)"
    },
    "preference": {
      "type": "string",
      "enum": ["shortest", "safe", "default"],
      "description": "shortest: 거리 shaping 추가 / safe: 회피 가중 / default: shaping 없음"
    },
    "interpretation_notes": {
      "type": "string",
      "description": "LLM이 모호한 표현을 어떻게 해석했는지 자기 보고"
    }
  }
}
```

**스키마 설계 원칙** (`schemas/design_rationale.md`로 별도 작성)

| 자연어 표현 | 매핑 위치 | 근거 |
|---|---|---|
| 목표 위치 ("~로 가") | `goal` (Reward 함수에 반영) | 원칙 2: 목적함수와 1:1 |
| 절대 회피 ("~는 절대 피해") | `obstacles` (환경 동역학) | 원칙 3: 환경에 위임 |
| 권장 회피 ("~는 피하는 게 좋아") | `soft_avoid` (Reward 페널티) | Hard vs Soft 구분 — 캡스톤의 핵심 질문 |
| 경로 선호 ("빠르게", "안전하게") | `preference` (Reward shaping 종류) | 원칙 2: 같은 목적, 다른 형태 |

---

### 3.2 Step 1: 동적 환경 구현

**파일**: `env.py`

```python
class NavGridEnv:
    def __init__(self,
                 grid_size: tuple,           # (height, width)
                 start: tuple,
                 goal: tuple,
                 obstacles: list = None,     # hard
                 soft_avoid: list = None,    # soft
                 preference: str = "default",
                 max_steps: int = 200):
        ...

    def reset(self) -> tuple: ...

    def step(self, action: int) -> tuple:
        """
        보상 규칙:
        - 목표 도달: +100, done=True
        - obstacle 진입: -100, reset (done=False지만 시작점으로)
        - soft_avoid 진입: -5 (그 외 -1과 합산하지 않음, -5 자체)
        - preference="shortest": 매 스텝 (이전 거리 - 현재 거리) * 0.5 추가
        - preference="safe": soft_avoid 인근 1셀에도 -2 페널티
        - 그 외: -1 (시간 페널티)
        - max_steps 초과: done=True (timeout)
        """

    def render(self, Q=None, save_path=None, title=""):
        """그리드 + obstacles(검정) + soft_avoid(회색) + goal(별) + greedy policy(화살표) 시각화"""
```

**검증 테스트** (구현 후 즉시 실행)
```python
# Sanity test
env = NavGridEnv(grid_size=(8,8), start=(0,0), goal=(7,7),
                 obstacles=[(3,3),(3,4)], soft_avoid=[(5,5)])
state = env.reset()
assert state == (0, 0)
ns, r, done, _ = env.step(1)  # 우
assert r == -1
```

---

### 3.3 Step 2: LLM 파서

**파일**: `nl_parser.py`

**전체 프롬프트** (`prompts/parser_prompt_v1.txt`로 저장하고 코드에서 로드)

```text
You are a strict JSON parser that converts Korean natural language navigation commands into a structured spec.

The grid is by default 8x8 with rows 0-7 (top to bottom) and columns 0-7 (left to right).
- (0, 0) = top-left corner
- (7, 7) = bottom-right corner
- (0, 7) = top-right corner
- (7, 0) = bottom-left corner

You must output ONLY a valid JSON object matching this schema:
{
  "grid_size": [height, width],
  "start": [row, col],
  "goal": [row, col],
  "obstacles": [[row, col], ...],
  "soft_avoid": [[row, col], ...],
  "preference": "shortest" | "safe" | "default",
  "interpretation_notes": "string"
}

Rules:
1. Default grid_size is [8, 8]. Default start is [0, 0].
2. Spatial expressions:
   - "왼쪽 위 모서리" → [0, 0]
   - "오른쪽 아래" → [7, 7]
   - "가운데" → [3, 3] or [4, 4] (choose [3, 3] and note in interpretation_notes)
3. Hard vs Soft avoidance:
   - "절대 피해", "장애물", "막혀있다" → obstacles
   - "피하는 게 좋아", "가급적 피해", "위험해" → soft_avoid
4. Preference:
   - "빠르게", "최단", "짧게" → "shortest"
   - "안전하게", "조심히" → "safe"
   - 명시 없음 → "default"
5. Ambiguous expressions:
   - "주변" → 1셀 반경으로 해석 (예: (4,4) 주변 → soft_avoid: [(3,4),(5,4),(4,3),(4,5)])
   - "근처" → 동일 (1셀 반경)
   - 항상 interpretation_notes에 어떻게 해석했는지 기록
6. If goal cannot be inferred, set goal to null and explain in interpretation_notes.

Output ONLY the JSON, no markdown code blocks, no additional text.

Examples:

Input: "오른쪽 아래 구석으로 가"
Output: {"grid_size":[8,8],"start":[0,0],"goal":[7,7],"obstacles":[],"soft_avoid":[],"preference":"default","interpretation_notes":"오른쪽 아래 구석을 (7,7)로 해석"}

Input: "(7,7)로 가는데 (4,4) 주변은 피해"
Output: {"grid_size":[8,8],"start":[0,0],"goal":[7,7],"obstacles":[],"soft_avoid":[[3,4],[5,4],[4,3],[4,5]],"preference":"default","interpretation_notes":"(4,4) 주변을 1셀 반경 4개 셀로 해석. 명시적 hard 키워드가 없어 soft_avoid로 분류"}

Now parse this command:
{COMMAND}
```

**파서 함수**
```python
from openai import OpenAI
import json
from jsonschema import validate

def parse_command(command: str, model: str = "gpt-4o") -> dict:
    """
    자연어 명령을 구조화된 JSON spec으로 변환.

    Returns:
        dict: schema_v1을 만족하는 spec
    Raises:
        ValueError: JSON 파싱 실패 또는 스키마 위반
    """
    client = OpenAI()

    with open("prompts/parser_prompt_v1.txt") as f:
        prompt_template = f.read()

    full_prompt = prompt_template.replace("{COMMAND}", command)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": full_prompt}]
    )

    raw_text = response.choices[0].message.content.strip()

    # JSON 파싱
    try:
        spec = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw_text}\nError: {e}")

    # 스키마 검증
    with open("schemas/command_schema_v1.json") as f:
        schema = json.load(f)
    validate(instance=spec, schema=schema)

    return spec
```

**테스트 케이스** (`test_cases/commands.json`)
```json
[
  {"id": "T01", "command": "오른쪽 아래 구석으로 가", "expected_category": "clear"},
  {"id": "T02", "command": "(7,7)로 가는데 (4,4) 주변은 피해", "expected_category": "constrained"},
  {"id": "T03", "command": "안전하게 목표까지", "expected_category": "ambiguous"},
  {"id": "T04", "command": "빠르게 가", "expected_category": "underspecified"},
  {"id": "T05", "command": "왼쪽 위로 가되 가운데 칸은 절대 피해", "expected_category": "clear_hard"},
  {"id": "T06", "command": "(0,7)로 가는 가장 짧은 길", "expected_category": "clear_pref"},
  {"id": "T07", "command": "오른쪽 위 구석으로 가되 (3,3)~(3,5) 라인은 위험해", "expected_category": "constrained_soft"},
  {"id": "T08", "command": "장애물 없이 목표까지", "expected_category": "underspecified"},
  {"id": "T09", "command": "(5,5)에서 출발해서 (0,0)까지 안전하게", "expected_category": "clear_with_start"},
  {"id": "T10", "command": "가운데로 가", "expected_category": "ambiguous"}
]
```

**파서 단위 테스트 스크립트** (`run_parser_tests.py`)
```python
results = []
for case in test_cases:
    try:
        spec = parse_command(case["command"])
        status = "PASS"
        notes = spec.get("interpretation_notes", "")
    except Exception as e:
        status = "FAIL"
        spec = None
        notes = str(e)
    results.append({
        "id": case["id"],
        "command": case["command"],
        "category": case["expected_category"],
        "status": status,
        "spec": spec,
        "notes": notes
    })

# results/parser_test_results.json으로 저장
# 마크다운 표로도 출력
```

**해석 템플릿** (Claude Code가 결과를 보고 자동 채움)
```markdown
## 파서 테스트 결과 분석

| ID | 명령 | 상태 | LLM 해석 | 평가 |
|---|---|---|---|---|
| T01 | ... | PASS | "..." | 의도 일치 |
| T03 | ... | PASS | "안전하게를 preference=safe로 해석" | 의도 일치 (소프트한 해석) |
| T08 | ... | ... | ... | **목표 미명시 — 어떻게 처리됐는가?** |

### 발견된 패턴
1. [LLM이 일관되게 잘 처리하는 표현 유형]
2. [모호성을 어떻게 자체 해소하는가]
3. [실패가 발생한 명령의 공통점]
```

---

### 3.4 Step 3: 환경 빌더

**파일**: `env_builder.py`

```python
def build_env(spec: dict) -> NavGridEnv:
    """
    JSON spec → NavGridEnv 인스턴스
    Spec validation은 parser에서 이미 됨.

    Edge case 처리:
    - goal이 obstacle 안에 있으면 → ValueError (의도-환경 불일치)
    - start == goal → ValueError
    - goal이 grid 밖 → ValueError
    """
    grid_size = tuple(spec["grid_size"])
    start = tuple(spec.get("start", [0, 0]))
    goal = tuple(spec["goal"])
    obstacles = [tuple(o) for o in spec.get("obstacles", [])]
    soft_avoid = [tuple(s) for s in spec.get("soft_avoid", [])]
    preference = spec.get("preference", "default")

    # Validation
    if goal in obstacles:
        raise ValueError(f"Goal {goal} is inside obstacles")
    if start == goal:
        raise ValueError("Start equals goal")
    if not (0 <= goal[0] < grid_size[0] and 0 <= goal[1] < grid_size[1]):
        raise ValueError(f"Goal {goal} is outside grid {grid_size}")

    return NavGridEnv(grid_size=grid_size, start=start, goal=goal,
                      obstacles=obstacles, soft_avoid=soft_avoid,
                      preference=preference)
```

---

### 3.5 Step 4: Q-Learning 에이전트 (Phase 1 재사용)

**파일**: `agent.py`

Phase 1의 `QLearningAgent`를 그대로 가져오되, state encoding을 (row, col) → flat index로 처리하는 헬퍼만 추가.

```python
class QLearningAgent:
    def __init__(self, grid_size, n_actions=4,
                 alpha=0.1, gamma=0.95, epsilon=0.1,
                 epsilon_decay=0.995, epsilon_min=0.01,
                 seed=42):
        self.grid_size = grid_size
        self.n_states = grid_size[0] * grid_size[1]
        self.n_actions = n_actions
        self.Q = np.zeros((self.n_states, n_actions))
        ...

    def _state_to_idx(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def act(self, state): ...
    def update(self, s, a, r, ns, done): ...
```

---

### 3.6 Step 5: 통합 파이프라인 + 실험

**파일**: `run_experiments.py`

```python
def run_pipeline(command: str,
                 episodes: int = 1500,
                 save_dir: str = None) -> dict:
    """
    End-to-end 실행:
    1. 명령 파싱
    2. 환경 빌드
    3. Q-learning 학습
    4. 결과 시각화 + 저장

    Returns:
        dict: {spec, learning_curve, final_policy, evaluation_metrics}
    """
    # 1. 파싱
    spec = parse_command(command)

    # 2. 빌드
    try:
        env = build_env(spec)
    except ValueError as e:
        return {"status": "build_failed", "error": str(e), "spec": spec}

    # 3. 학습
    agent = QLearningAgent(grid_size=tuple(spec["grid_size"]))
    rewards_per_episode = train(agent, env, episodes)

    # 4. 평가 (greedy 정책으로 10회 rollout 평균)
    metrics = evaluate(agent, env, n_rollouts=10)

    # 5. 시각화
    if save_dir:
        plot_learning_curve(rewards_per_episode, f"{save_dir}/learning_curve.png")
        env.render(Q=agent.Q, save_path=f"{save_dir}/policy.png",
                   title=f"Command: {command}")
        with open(f"{save_dir}/spec.json", "w") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "command": command,
        "spec": spec,
        "metrics": metrics,
        "learning_curve": rewards_per_episode
    }
```

**실험 정의**

```python
EXPERIMENTS = [
    {
        "id": "A_clear",
        "command": "오른쪽 아래 구석으로 가는 가장 짧은 길",
        "hypothesis": "직선에 가까운 단순 경로 학습. preference=shortest로 거리 shaping이 들어가 학습 속도 빠름.",
        "success_criteria": "수렴 후 평균 경로 길이 ≤ 16 (8x8 그리드의 최단)"
    },
    {
        "id": "B_constrained",
        "command": "(7,7)로 가되 (3,3), (3,4), (4,3), (4,4) 영역은 절대 피해",
        "hypothesis": "장애물을 우회하는 우회 경로 학습. obstacles로 hard 처리되어 절대 진입 안 함.",
        "success_criteria": "학습된 경로에 obstacle 셀이 0번 등장"
    },
    {
        "id": "C_ambiguous",
        "command": "안전하게 (7,7)까지",
        "hypothesis": "LLM이 '안전하게'를 preference=safe로 해석. soft_avoid는 비어있을 가능성 — 이때 'safe'가 무의미해짐.",
        "success_criteria": "LLM의 interpretation_notes에 '안전하게' 해석 명시 + 학습 자체는 수렴"
    },
    {
        "id": "D_failure",
        "command": "빠르게 가",
        "hypothesis": "목표 미명시. LLM이 (a) 임의로 목표 설정 또는 (b) goal=null 반환. 후자면 build_env에서 ValueError.",
        "success_criteria": "실패가 어느 단계(파싱/빌드/학습)에서 발생했는지 명확히 식별"
    }
]
```

**자동 실행 스크립트**
```python
all_results = []
for exp in EXPERIMENTS:
    save_dir = f"results/exp_{exp['id']}"
    os.makedirs(save_dir, exist_ok=True)
    result = run_pipeline(exp["command"], save_dir=save_dir)
    result["hypothesis"] = exp["hypothesis"]
    result["success_criteria"] = exp["success_criteria"]
    all_results.append(result)

with open("results/all_experiments.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
```

---

### 3.7 Step 6: 결과 해석 자동 생성

**해석 템플릿** (Claude Code가 모든 실험 완료 후 자동 작성)

각 실험마다 `results/exp_X/interpretation.md` 생성:

```markdown
# 실험 [ID]: [명령]

## 입력
- 자연어 명령: "..."
- 가설: ...
- 성공 기준: ...

## LLM 파싱 결과
```json
[spec.json 내용]
```

**해석 노트**: [LLM의 interpretation_notes 인용 + Claude Code의 평가 1줄]

## 학습 결과
- 수렴 에피소드: [측정]
- 최종 평균 보상: [측정]
- greedy rollout 평균 경로 길이: [측정]
- obstacle 진입 횟수 (rollout 중): [측정]

## 가설 vs 결과
[가설이 맞았는가, 틀렸는가, 부분적으로 맞았는가 — 3-5줄]

## 발견된 한계
[이 실험에서 시스템이 어떻게 깨졌는가 / 어디서 의도-구현 갭이 발생했는가]
```

---

### 3.8 Step 7: 실패 분류표 (이 프로젝트의 핵심 자산)

**파일**: `results/failure_taxonomy.md`

Claude Code는 4개 실험 결과를 종합하여 다음 표를 자동 작성:

```markdown
# NL → MDP 변환 시스템의 실패 지점 분류

## 분류 체계

| 실패 유형 | 발생 위치 | 발견된 사례 | 캡스톤에서의 대응 방향 |
|---|---|---|---|
| **파싱 실패** | LLM 단계 | [실험 결과에서 발견] | 스키마 강제 + few-shot 예시 추가 |
| **의도 왜곡** | LLM 단계 | "주변"을 1셀로 해석했지만 사용자는 3셀 의도 | 명령어 사전 또는 사용자 피드백 루프 |
| **목표 미명시** | LLM 단계 | T04 "빠르게 가" | goal=null 반환 + 사용자에게 재질의 |
| **의도-환경 불일치** | Builder 단계 | goal이 obstacle 안 | spec validation layer (현재 구현됨) |
| **학습 실패** | RL 단계 | sparse reward + 큰 grid에서 미수렴 | reward shaping 자동화 또는 함수 근사 |

## 빈도 통계
[4개 실험 중 각 유형이 몇 번 발생했는가]

## 가장 위험한 유형
[빈도 + 심각도 종합 1-2개 선정 + 근거]
```

---

### 3.9 Step 8: 최종 README — Capstone Bridge

**파일**: `README.md`

```markdown
# NL-Conditioned Q-Learning: 자연어 명령 정찰

## 프로젝트 정체성
이 프로젝트는 **캡스톤이 아니다**. "자연어 → MDP 변환 파이프라인이 어디서 작동하고 어디서 깨지는가"를 가장 작은 환경에서 매핑한 정찰 작업이다.

## 시스템 구조
[architecture.md 내용 요약]

## 핵심 발견 3가지

### 1. 작동하는 지점
- 명확한 좌표 + 명확한 hard 제약 → end-to-end 파이프라인이 의도된 정책으로 수렴
- LLM은 한국어 공간 표현("왼쪽 위", "오른쪽 아래")을 일관되게 좌표로 매핑

### 2. 깨지는 지점
[failure_taxonomy.md의 핵심 행 인용]

### 3. 가장 큰 의외
[실험에서 발견한 예상 외의 결과 1개 — 예: "안전하게"의 LLM 해석이 어떠했는가]

## From NL-Grid to Capstone

### 확장 경로
| 차원 | 현재 (정찰) | 캡스톤 (예상) |
|---|---|---|
| State | (row, col) | 로봇 상태 + 라이다 관측 |
| Action | 4방향 이산 | 연속 속도 명령 |
| Q 표현 | tabular | 신경망 (DQN/PPO) |
| 환경 | 8×8 grid | PyBullet 2D 평면 |
| 명령 스키마 | v1 (위치 + obstacle) | v2 (task-space + 시간 제약) |

### 재사용 가능한 자산
- **LLM 파서 + 프롬프트**: 스키마만 v2로 확장하면 그대로 사용
- **명령-환경 빌더 패턴**: spec → env 매핑 로직 재사용
- **실패 분류표**: 캡스톤의 한계 명시 슬라이드의 골격

### 캡스톤에서 우선 해결할 문제
[failure_taxonomy.md의 "가장 위험한 유형"을 인용 + 그것을 해결하는 게 캡스톤의 차별점이 되는 이유]

## 실행 방법
```bash
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
python run_parser_tests.py     # LLM 파서 단위 테스트
python run_experiments.py      # 4개 실험 자동 실행
```
```

---

## 4. 전체 작업 체크리스트 (Claude Code용)

### Phase 1 (압축)
- [ ] `cliff-walking/env.py` 작성 + sanity test
- [ ] `cliff-walking/agents.py` 작성 (SARSA + Q-learning)
- [ ] `cliff-walking/run_experiments.py`로 6개 실험 실행
- [ ] 학습 곡선, 정책 화살표 시각화 저장
- [ ] README 자동 생성

### Phase 2 (메인)
- [ ] `nl-conditioned-grid/schemas/command_schema_v1.json` 작성
- [ ] `schemas/design_rationale.md` 작성
- [ ] `env.py` (NavGridEnv) 작성 + sanity test
- [ ] `prompts/parser_prompt_v1.txt` 작성
- [ ] `nl_parser.py` 작성
- [ ] `test_cases/commands.json` 작성 (10개)
- [ ] `run_parser_tests.py` 실행 → 결과 분석 마크다운 자동 생성
- [ ] `env_builder.py` 작성 + edge case test
- [ ] `agent.py` 작성 (Phase 1 재사용 + state encoding)
- [ ] `run_experiments.py`로 4개 실험 실행
- [ ] 각 실험 `interpretation.md` 자동 생성
- [ ] `failure_taxonomy.md` 자동 생성
- [ ] 최종 `README.md` (Capstone Bridge 포함) 자동 생성

---

## 5. 매일 작성할 학습 일지 (사용자가 작성)

매일 작업 끝에 `daily_log.md`에 다음 형식으로 추가:

```markdown
## YYYY-MM-DD (작업 N시간)
- 완료한 단계: [Step 번호]
- 발견한 것 (성공이든 실패든): [한 문장]
- 의미: [캡스톤과의 연결 1줄]
- 내일 할 것: [한 문장]
```

이 일지가 최종보고서의 "주차별 진행 경과" 섹션의 1차 자료가 된다.

---

## 6. 사용자(승재)가 직접 검증할 부분

Claude Code에 모든 걸 맡기지 말 것. 다음 두 가지는 직접 손으로 짜고 비교해야 한다 (메모리상 본인의 학습 원칙):

1. **Q-learning 업데이트 식**: `agents.py`의 `update()` 메서드를 보기 전에 직접 종이에 쓴다. Module 5에서 정리한 벨만 업데이트와 비교.
2. **명령 스키마 v1**: Claude Code가 제안한 스키마를 그대로 쓰지 말고, 본인이 보고서에 도출한 MDP 설계 원칙 3가지에 비추어 검토. 특히 "soft_avoid를 Reward에 넣을지 환경 동역학에 넣을지"는 본인 판단.

---

## 7. 예상 일정

| 일자 | 작업 | 산출물 |
|---|---|---|
| Day 1 | Phase 1 환경 + SARSA | env.py, sarsa working |
| Day 2 | Phase 1 Q-learning + 실험 + 정리 | results/, README |
| Day 3 | Phase 2 스키마 + 환경 | schema, NavGridEnv |
| Day 4 | LLM 파서 + 단위 테스트 | nl_parser.py, parser_test_results |
| Day 5 | 빌더 + 통합 + 실험 A,B | exp_A, exp_B 결과 |
| Day 6 | 실험 C,D + 분류표 + Capstone Bridge README | 최종 산출물 |

총 6일, 일평균 2-3시간 작업 가정.

---

## 8. Claude Code에게 전달할 때 추가할 컨텍스트

이 명세서를 Claude Code에게 던질 때 다음 한 줄을 함께 전달:

> "이 명세서대로 구현해줘. Phase 1은 압축적으로, Phase 2는 디테일하게. 각 단계 완료 후 sanity test를 자동 실행해서 결과를 보여줘. 모든 실험 완료 후 failure_taxonomy.md와 최종 README를 자동 작성해줘. 다만 agents.py의 Q-learning update 메서드는 # TODO: implement yourself 주석만 남기고 비워둬 (사용자가 직접 채움)."
