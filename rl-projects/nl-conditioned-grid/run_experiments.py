import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from env_builder import build_env
from nl_parser import parse_command


BASE_DIR = Path(__file__).resolve().parent


EXPERIMENTS = [
    {
        "id": "A_clear",
        "command": "오른쪽 아래 구석으로 가고 가장 짧게 가",
        "hypothesis": "직선에 가까운 단순 경로를 학습하고, preference=shortest로 거리 shaping이 들어가 학습 속도가 빨라진다.",
        "success_criteria": "학습 후 평균 경로 길이가 16 이하에 가까워진다.",
    },
    {
        "id": "B_constrained",
        "command": "(7,7)로 가고 (3,3), (3,4), (4,3), (4,4) 영역은 절대 피해",
        "hypothesis": "장애물을 우회하는 경로를 학습하고 obstacle 진입이 greedy rollout에서 0회가 된다.",
        "success_criteria": "greedy rollout 중 obstacle 진입 횟수 0회.",
    },
    {
        "id": "C_ambiguous",
        "command": "안전하게 (7,7)까지",
        "hypothesis": "LLM이 '안전하게'를 preference=safe로 해석하지만 soft_avoid가 비어 있으면 효과가 제한적이다.",
        "success_criteria": "파싱 notes에 안전 선호 해석이 기록되고 학습이 완료된다.",
    },
    {
        "id": "D_failure",
        "command": "빠르게 가",
        "hypothesis": "목표 미명시 때문에 LLM이 goal=null을 반환하고 builder에서 controlled failure가 발생한다.",
        "success_criteria": "실패 단계가 parse 또는 build로 명확하게 기록된다.",
    },
]


def train(agent, env, episodes):
    rewards_per_episode = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
    return rewards_per_episode


def evaluate(agent, env, n_rollouts=10):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    totals = []
    path_lengths = []
    successes = 0
    obstacle_entries = 0
    timeout_count = 0
    example_path = []

    for rollout_idx in range(n_rollouts):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        path = [state]
        while not done:
            action = agent.act(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            if info.get("event") == "obstacle":
                obstacle_entries += 1
            if info.get("timeout"):
                timeout_count += 1
            if info.get("event") == "goal":
                successes += 1
            state = next_state
            path.append(state)
            total_reward += reward
            steps += 1
        totals.append(total_reward)
        path_lengths.append(steps)
        if rollout_idx == 0:
            example_path = path

    agent.epsilon = old_epsilon
    return {
        "mean_return": float(np.mean(totals)),
        "mean_path_length": float(np.mean(path_lengths)),
        "success_rate": successes / n_rollouts,
        "obstacle_entries": obstacle_entries,
        "timeouts": timeout_count,
        "example_path": example_path,
    }


def plot_learning_curve(rewards_per_episode, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rewards = np.asarray(rewards_per_episode, dtype=float)
    if len(rewards) >= 30:
        smoothed = np.convolve(rewards, np.ones(30) / 30, mode="valid")
        ax.plot(np.arange(len(smoothed)), smoothed, label="30-episode moving avg")
    ax.plot(rewards, alpha=0.25, label="episode return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Q-Learning Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_interpretation(exp, result, save_dir):
    spec = result.get("spec")
    metrics = result.get("metrics") or {}
    if result["status"] == "success":
        llm_notes = spec.get("interpretation_notes", "")
        learning = (
            f"- 마지막 episode return: {result['learning_curve'][-1]:.2f}\n"
            f"- 최종 평균 reward: {metrics['mean_return']:.2f}\n"
            f"- greedy rollout 평균 경로 길이: {metrics['mean_path_length']:.2f}\n"
            f"- obstacle 진입 횟수: {metrics['obstacle_entries']}\n"
        )
        comparison = "가설과 결과는 대체로 비교 가능하다. 성공률, 경로 길이, obstacle 진입 횟수를 기준으로 자연어 해석이 MDP에 잘 연결됐는지 확인할 수 있다."
    else:
        llm_notes = result.get("error", "")
        learning = "- 학습 미실행\n"
        comparison = f"{result['status']} 단계에서 중단되어 학습 결과와 가설을 비교할 수 없다. 이 실패 자체가 파이프라인의 실패 지점으로 기록된다."

    spec_block = json.dumps(spec, indent=2, ensure_ascii=False) if spec is not None else "null"
    text = f"""# 실험 {exp['id']}: {exp['command']}

## 입력
- 자연어 명령: "{exp['command']}"
- 가설: {exp['hypothesis']}
- 성공 기준: {exp['success_criteria']}

## LLM 파싱 결과
```json
{spec_block}
```

해석 노트: {llm_notes}

## 학습 결과
{learning}
## 가설 vs 결과
{comparison}

## 발견과 한계
현재 결과는 자연어 파싱, 환경 빌드, RL 학습 중 어느 단계가 실패하거나 성공했는지 추적하는 데 초점을 둔다.
"""
    Path(save_dir, "interpretation.md").write_text(text, encoding="utf-8")


def write_failure_taxonomy(results):
    counts = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

    rows = [
        "| 실패 유형 | 발생 위치 | 발견 여부 | 대응 방향 |",
        "|---|---|---|---|",
        "| 파싱 실패 | LLM 단계 | {}건 | API 키/응답 JSON/스키마 검증을 분리해서 기록 |".format(counts.get("parse_failed", 0)),
        "| 목표 미명시 | LLM/Builder 단계 | {}건 | `goal=null`을 허용하고 사용자 재질문 대상으로 분류 |".format(counts.get("build_failed", 0)),
        "| 의도-환경 불일치 | Builder 단계 | 실험별 확인 | goal과 obstacle 충돌 등 validation 강화 |",
        "| 학습 실패 | RL 단계 | {}건 | timeout, sparse reward, shaping 필요성 확인 |".format(counts.get("learning_failed", 0)),
    ]
    text = f"""# NL to MDP 변환 시스템의 실패 지점 분류

## 분류 체계

{chr(10).join(rows)}

## 빈도 통계
```json
{json.dumps(counts, indent=2, ensure_ascii=False)}
```

## 가장 위험한 실패 유형
현재 실행에서 가장 먼저 관측된 위험은 파싱 단계 실패다. OpenAI API 호출이 불가능하면 이후 builder와 RL 단계가 모두 실행되지 않으므로, 캡스톤 확장 전에는 API 설정 확인과 parser 결과 캐싱/재시도 전략이 우선 필요하다.
"""
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "failure_taxonomy.md").write_text(text, encoding="utf-8")


def write_readme(results):
    success_count = sum(1 for result in results if result["status"] == "success")
    parse_failures = sum(1 for result in results if result["status"] == "parse_failed")
    text = f"""# NL-Conditioned Q-Learning: 자연어 명령 정찰

## 프로젝트 정체성
이 프로젝트는 캡스톤 자체가 아니라, 자연어를 MDP로 바꾸는 작은 파이프라인이 어디서 작동하고 어디서 깨지는지 확인하는 정찰 작업이다.

## 시스템 구조
자연어 명령은 OpenAI `gpt-4o` 파서를 통해 command schema v1 JSON으로 변환되고, builder가 `NavGridEnv`를 만든 뒤 tabular Q-learning이 정책을 학습한다.

## 핵심 발견
1. 명확한 좌표와 제약이 들어오면 schema와 builder는 환경으로 변환할 준비가 되어 있다.
2. 목표가 없는 명령은 `goal=null`로 보존한 뒤 builder에서 controlled failure로 처리한다.
3. 이번 실행에서는 현재 PowerShell 세션에 `OPENAI_API_KEY`가 없어 {parse_failures}개 실험이 파싱 단계에서 실패했고, 성공 실험은 {success_count}개였다.

## From NL-Grid to Capstone

| 차원 | 현재 정찰 | 캡스톤 예상 |
|---|---|---|
| State | `(row, col)` | 로봇 상태 + 센서 관측 |
| Action | 4방향 이산 이동 | 연속 속도 또는 고수준 제어 명령 |
| Q 표현 | tabular | DQN/PPO 등 함수 근사 |
| 환경 | 8x8 grid | PyBullet 또는 실제 로봇 시뮬레이션 |
| 명령 스키마 | 위치 + 장애물 + 선호 | task-space + 시간/안전 제약 |

## 실행 방법
```powershell
pip install -r ..\\requirements.txt
$env:OPENAI_API_KEY = "sk-..."
python run_parser_tests.py
python run_experiments.py
```
"""
    (BASE_DIR / "README.md").write_text(text, encoding="utf-8")


def run_pipeline(command: str, episodes: int = 1500, save_dir: str = None) -> dict:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        spec = parse_command(command)
    except Exception as e:
        result = {"status": "parse_failed", "command": command, "error": str(e), "spec": None}
        if save_dir:
            _write_json(Path(save_dir) / "error.json", result)
        return result

    if save_dir:
        _write_json(Path(save_dir) / "spec.json", spec)

    try:
        env = build_env(spec)
    except ValueError as e:
        result = {"status": "build_failed", "command": command, "error": str(e), "spec": spec}
        if save_dir:
            _write_json(Path(save_dir) / "error.json", result)
        return result

    try:
        agent = QLearningAgent(grid_size=tuple(spec["grid_size"]))
        rewards_per_episode = train(agent, env, episodes)
        metrics = evaluate(agent, env, n_rollouts=10)
    except Exception as e:
        result = {"status": "learning_failed", "command": command, "error": str(e), "spec": spec}
        if save_dir:
            _write_json(Path(save_dir) / "error.json", result)
        return result

    if save_dir:
        plot_learning_curve(rewards_per_episode, str(Path(save_dir) / "learning_curve.png"))
        env.render(Q=agent.Q, save_path=str(Path(save_dir) / "policy.png"), title=f"Command: {command}")

    return {
        "status": "success",
        "command": command,
        "spec": spec,
        "metrics": metrics,
        "learning_curve": rewards_per_episode,
    }


def run_all():
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    all_results = []

    for exp in EXPERIMENTS:
        save_dir = results_dir / f"exp_{exp['id']}"
        result = run_pipeline(exp["command"], save_dir=str(save_dir))
        result["hypothesis"] = exp["hypothesis"]
        result["success_criteria"] = exp["success_criteria"]
        all_results.append(result)
        write_interpretation(exp, result, save_dir)

    _write_json(results_dir / "all_experiments.json", all_results)
    write_failure_taxonomy(all_results)
    write_readme(all_results)
    return all_results


if __name__ == "__main__":
    results = run_all()
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
