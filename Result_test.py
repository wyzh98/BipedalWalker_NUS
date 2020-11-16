import gym
import numpy as np
from PPO_lstm import ENV, PPO

env = gym.make(ENV)
ppo = PPO(env, './', gpu=True)
while True:
    s = env.reset()
    ppo.restore_model('./20201113-220850/')
    ep_r, ep_t = 0, 0
    while True:
        env.render()
        a, v = ppo.evaluate_state(s, training=False)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        s, r, terminal, _ = env.step(a)
        ep_r += r
        ep_t += 1
        if terminal:
            f=open('ResultsTest.txt', 'a')
            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
            f.write("Reward: " +str(ep_r), '| Steps: ' + str(ep_r) + '\n')
            f.close()
            break