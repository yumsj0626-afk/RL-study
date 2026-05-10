from env import CliffWalkingEnv
import numpy as np

env = CliffWalkingEnv()
Q = np.zeros((env.height * env.width, env.n_actions))
env.render_policy(Q, save_path="policy.png")
print("policy.png 생성 완료")