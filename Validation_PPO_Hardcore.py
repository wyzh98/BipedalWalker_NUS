import gym
import numpy as np
from models import PPO_HC

env = gym.make('BipedalWalkerHardcore-v3')
ppo = PPO_HC(env, None, gpu=True)
ppo.restore_model(r'./events/ppo_hardcore/model.ckpt-10500')
while True:
    s = env.reset()
    ep_r, ep_t = 0, 0
    while True:
        env.render()
        a, v = ppo.choose_state(s, training=False)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        s, r, terminal, _ = env.step(a)
        ep_r += r
        ep_t += 1
        if terminal:
            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
            break