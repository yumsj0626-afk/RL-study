import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import QLearningAgent, SARSAAgent
from env import CliffWalkingEnv


EPISODES = 500
EPSILONS = [0.01, 0.1, 0.3]
SEED = 42
MAX_STEPS_PER_EPISODE = 1000


def state_to_idx(state, width):
    return state[0] * width + state[1]


def train_sarsa(env, agent, episodes=EPISODES):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        s = state_to_idx(state, env.width)
        action = agent.act(s)
        done = False
        total_reward = 0

        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            next_state, reward, done, _ = env.step(action)
            ns = state_to_idx(next_state, env.width)
            next_action = agent.act(ns)
            agent.update(s, action, reward, ns, next_action, done)
            s, action = ns, next_action
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
    return rewards


def train_qlearning(env, agent, episodes=EPISODES):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            s = state_to_idx(state, env.width)
            action = agent.act(s)
            next_state, reward, done, _ = env.step(action)
            ns = state_to_idx(next_state, env.width)
            agent.update(s, action, reward, ns, done)
            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
    return rewards


def moving_average(values, window=20):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        rewards = result["episode_rewards"]
        smoothed = moving_average(rewards)
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, label=f"{result['algorithm']} eps={result['epsilon']}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (20-episode moving avg)")
    ax.set_title("Cliff Walking Learning Curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def summarize(results):
    return {
        f"{result['algorithm']}_eps_{result['epsilon']}": {
            "mean_return_last_100": float(np.mean(result["episode_rewards"][-100:])),
            "min_return_last_100": float(np.min(result["episode_rewards"][-100:])),
            "max_return_last_100": float(np.max(result["episode_rewards"][-100:])),
            "note": result.get("note", ""),
        }
        for result in results
    }


def write_readme(summary, save_path):
    sarsa_eps01 = summary["SARSA_eps_0.1"]["mean_return_last_100"]
    q_eps01 = summary["Q-learning_eps_0.1"]["mean_return_last_100"]
    q_eps03 = summary["Q-learning_eps_0.3"]["mean_return_last_100"]
    text = f"""# Cliff Walking: SARSA vs Q-Learning

## 핵심 발견
1. epsilon=0.1에서 SARSA의 마지막 100 episode 평균 return은 {sarsa_eps01:.2f}입니다.
2. Q-learning은 off-policy Bellman backup을 사용하여 경험에서 학습하며, epsilon=0.1 평균 return은 {q_eps01:.2f}입니다.
3. epsilon=0.3에서도 Q-learning 평균 return은 {q_eps03:.2f}입니다.

## 왜 SARSA가 더 긴 경로를 선호하는가
SARSA는 on-policy 방법이라 epsilon-greedy로 실제 실행되는 탐험 행동의 비용까지 업데이트에 반영한다. 절벽 근처의 최단 경로는 한 번의 탐험 행동으로 큰 손실을 낼 수 있으므로, SARSA는 그 위험을 Q 값에 포함해 더 안전한 우회 경로를 선호하는 경향이 있다. 반면 Q-learning은 다음 상태에서 항상 greedy 행동을 한다고 가정하는 off-policy 업데이트라, 구현하면 절벽 바로 위의 더 짧은 경로를 더 쉽게 선택한다.

## 산출물
- `results/learning_curves.png`
- `results/policy_sarsa_eps01.png`
- `results/policy_qlearning_eps01.png`
- `results/summary.json`
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)


def run_all():
    os.makedirs("results", exist_ok=True)
    results = []
    policies = {}

    for epsilon in EPSILONS:
        env = CliffWalkingEnv()
        sarsa = SARSAAgent(
            n_states=env.height * env.width,
            n_actions=env.n_actions,
            epsilon=epsilon,
            seed=SEED,
        )
        sarsa_rewards = train_sarsa(env, sarsa)
        results.append(
            {
                "algorithm": "SARSA",
                "epsilon": epsilon,
                "episode_rewards": sarsa_rewards,
            }
        )
        if epsilon == 0.1:
            policies["sarsa"] = sarsa.Q.copy()

        env = CliffWalkingEnv()
        q_agent = QLearningAgent(
            n_states=env.height * env.width,
            n_actions=env.n_actions,
            epsilon=epsilon,
            seed=SEED,
        )
        q_rewards = train_qlearning(env, q_agent)
        results.append(
            {
                "algorithm": "Q-learning",
                "epsilon": epsilon,
                "episode_rewards": q_rewards,
                "note": "Q-learning update uses off-policy Bellman backup with max action-value.",
            }
        )
        if epsilon == 0.1:
            policies["qlearning"] = q_agent.Q.copy()

    plot_learning_curves(results, "results/learning_curves.png")
    env = CliffWalkingEnv()
    env.render_policy(policies["sarsa"], "results/policy_sarsa_eps01.png")
    env.render_policy(policies["qlearning"], "results/policy_qlearning_eps01.png")

    summary = summarize(results)
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_readme(summary, "README.md")
    return summary


if __name__ == "__main__":
    summary = run_all()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
