import tensorflow as tf
import numpy as np
import gym
import os
from datetime import datetime
from models import PPO
from utils import discount, RunningStats, make_gif

GAMMA = 0.99
LAMBDA = 0.95  # advantage estimation discounting factor (lambda in the paper)
STEPBATCH = 8192
IS_HARDCORE = False
IS_RESTORED = False
IS_OUTPUTGIF = True
if IS_RESTORED:
    MODEL_PATH = r'./xxx/model.ckpt-xxx'
else:
    MODEL_PATH = None
if IS_HARDCORE:
    EPISODE_MAX = 100000
    ENV = 'BipedalWalkerHardcore-v3'
    RESULT_ROOT_PATH = './log_hc/'
else:
    EPISODE_MAX = 10000
    ENV = 'BipedalWalker-v3'
    RESULT_ROOT_PATH = './log/'

TIMESTAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
SUMMARY_DIR = os.path.join(RESULT_ROOT_PATH, TIMESTAMP)


def run():
    env = gym.make(ENV)
    ppo = PPO(env, SUMMARY_DIR, gpu=True)
    rew_list = []

    t, terminal = 0, False
    rollout_s, rollout_a, rollout_r, rollout_v, rollout_terminal = [], [], [], [], []
    stats_r = RunningStats()
    if MODEL_PATH is not None:
        ppo.restore_model(MODEL_PATH)
    for episode in range(EPISODE_MAX):
        s = env.reset()
        ep_r, ep_t, ep_a = 0, 0, []
        saveGIF = False
        while True:
            if episode % 500 == 0 and IS_OUTPUTGIF and saveGIF == False:
                saveGIF = True
                episode_frames = [env.render(mode='rgb_array')]

            a, v = ppo.choose_state(s, training=True)
            # Update ppo
            if t == STEPBATCH:  # or (terminal and t < BATCH):
                # Normalise rewards
                rewards = np.array(rollout_r)
                stats_r.update(rewards)
                rewards = np.clip(rewards / stats_r.std, -10, 10)

                v_final = [v * (1 - terminal)]  # vf = 0 if terminal, otherwise use the predicted vf
                values = np.array(rollout_v + v_final)
                terminals = np.array(rollout_terminal + [terminal])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(rollout_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)
                bs, ba, br, badv = np.reshape(rollout_s, (t,) + ppo.s_dim), np.vstack(rollout_a), \
                                   np.vstack(returns), np.vstack(advantage)

                graph_summary = ppo.train(bs, ba, br, badv)
                rollout_s, rollout_a, rollout_r, rollout_v, rollout_terminal = [], [], [], [], []
                t = 0
            rollout_s.append(s)
            rollout_a.append(a)
            rollout_v.append(v)
            rollout_terminal.append(terminal)
            ep_a.append(a)

            a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, _ = env.step(a)
            rollout_r.append(r)
            ep_r += r
            ep_t += 1
            t += 1

            if saveGIF and t % 2 == 0:
                a = env.render(mode='rgb_array')
                episode_frames.append(a)

            if terminal:
                print('Episode: %i' % episode, '| Reward: %.2f' % ep_r, '| Steps: %i' % ep_t)
                rew_list.append(ep_r)
                if episode % 100 == 0:
                    print('Mean reward of the past 100 episodes: ', str(np.mean(rew_list[-100:])))

                worker_summary = tf.Summary()
                worker_summary.value.add(tag="Perf/Reward", simple_value=ep_r)
                worker_summary.value.add(tag="Perf/Length", simple_value=ep_t)
                ppo.writer.add_summary(worker_summary, episode)
                try:
                    ppo.writer.add_summary(graph_summary, episode)
                except NameError:
                    pass
                ppo.writer.flush()

                if saveGIF:
                    images = np.array(episode_frames)
                    make_gif(images, '{}/episode_{}.gif'.format(SUMMARY_DIR, episode))

                # Save the model
                if episode % 500 == 0 and episode > 0:
                    path = ppo.save_model(SUMMARY_DIR, episode)
                    print('Saved model at episode', episode, 'in', path)

                break


if __name__ == '__main__':
    run()
