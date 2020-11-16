import tensorflow as tf
import numpy as np
import gym
import os
import scipy
import matplotlib.pyplot as plt

from datetime import datetime
from time import time

EPISODE_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95  # advantage estimation discounting factor (lambda in the paper)
LR = 1e-4
BATCH = 4096
MINIBATCH = 32
EPOCHS = 4
EPSILON = 0.2
L2_REG = 1e-3
EPS_LEN = 2e5

IS_HARDCORE = False
MODEL_PATH = None
RESULT_ROOT_PATH = "./log/"
ENV = 'BipedalWalker-v3'

if IS_HARDCORE:
    RESULT_ROOT_PATH = "./log_hc/"
    ENV = 'BipedalWalkerHardcore-v3'


def discount(x, gamma, terminal_array=None):
    if terminal_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]


class RunningStats(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count


class PPO:
    def __init__(self, environment, summary_dir="./", gpu=True):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': gpu})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.summary_dir = summary_dir

        ########## Arrange the environment
        self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.shape[0]
        self.a_bound = (environment.action_space.high - environment.action_space.low) / 2
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        ########## Open tensors for data
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None, self.s_dim[0]], 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        ########## Build synthetic q
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()
        ########## Call the nets
        pi_old, pi_old_params = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params = self._build_anet(batch["state"], 'pi')
        pi_eval, _ = self._build_anet(self.state, 'pi', reuse=True)

        vf_old, vf_old_params = self._build_cnet(batch["state"], "oldvf")
        self.vf, vf_params = self._build_cnet(batch["state"], "vf")
        self.vf_eval, _ = self._build_cnet(self.state, 'vf', reuse=True)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mode()  # Used mode for discrete case. Mode should equal mean in continuous
        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver()
        ################ LOSS
        epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, EPS_LEN, 0.01, power=0.5)

        # Use floor functions for the probabilities to prevent NaNs when prob = 0
        ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = batch["advantage"] * ratio
        surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        loss_pg = - 2.0 * tf.reduce_mean(tf.minimum(surr1, surr2))

        # Sometimes values clipping helps, sometimes just using raw residuals is better ¯\_(ツ)_/¯
        # clipped_value_estimate = vf_old + tf.clip_by_value(self.vf - vf_old, -epsilon_decay, epsilon_decay)
        # loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
        # loss_vf2 = tf.squared_difference(self.vf, batch["rewards"])
        # loss_vf = 0.5 * tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2))
        loss_vf = 0.5 * tf.reduce_mean(tf.square(self.vf - batch["rewards"]))

        entropy = pi.entropy()
        loss_entropy = - 0.01 * tf.reduce_mean(entropy)

        loss = loss_pg + loss_vf + loss_entropy

        tf.summary.scalar("Loss/Policy", loss_pg)
        tf.summary.scalar("Loss/Value", loss_vf)
        # tf.summary.scalar("Loss/Vf1", loss_vf1)
        # tf.summary.scalar("Loss/Vf2", loss_vf2)
        tf.summary.scalar("Loss/Entropy", loss_entropy)
        tf.summary.scalar("Loss/Total", loss)
        tf.summary.scalar("Epsilon", epsilon_decay)

        ############### TRAIN
        opt = tf.train.AdamOptimizer(LR)
        self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)

        ############### Update thetas
        self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
        self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]

        ############### TensorBoard
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("value", tf.reduce_mean(self.vf))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))
        tf.summary.scalar("sigma", tf.reduce_mean(pi.stddev()))

        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    ############# Models
    def _build_anet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            layer_1 = tf.layers.dense(state_in, 512, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            layer_2 = tf.layers.dense(layer_1, 512, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(layer_2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")

            log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
            dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), 0.))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params

    def _build_cnet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 512, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, 512, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")
            vf = tf.layers.dense(l2, 1, kernel_regularizer=w_reg, name="vf_output")

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params

    ########## Update function
    def train(self, s, a, r, adv):
        start = time()
        e_time = []

        self.sess.run([self.update_pi_old_op, self.update_vf_old_op, self.iterator.initializer],
                      feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})

        while True:
            try:
                summary, step, _ = self.sess.run([self.summarise, self.global_step, self.train_op])
            except tf.errors.OutOfRangeError:
                break
        print("Trained in %.3fs.Global step %i" % (time() - start, step))
        return summary

    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, "model.ckpt"), global_step=step)
        return save_path

    def restore_model(self, model_path):
        self.saver.restore(self.sess, os.path.join(model_path, "model.ckpt"))
        print("Model restored from", model_path)

    def evaluate_state(self, state, training=True):
        if training:
            action, value = self.sess.run([self.sample_op, self.vf_eval], {self.state: state[np.newaxis, :]})
        else:
            action, value = self.sess.run([self.eval_action, self.vf_eval], {self.state: state[np.newaxis, :]})
        return action[0], np.squeeze(value)


if __name__ == '__main__':
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(RESULT_ROOT_PATH, TIMESTAMP)
    env = gym.make(ENV)

    ppo = PPO(env, SUMMARY_DIR, gpu=True)
    rew_list = []
    mean_100 = []
    if MODEL_PATH is not None:
        ppo.restore_model(MODEL_PATH)

    t, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
    rolling_r = RunningStats()

    for episode in range(EPISODE_MAX):

        s = env.reset()
        ep_r, ep_t, ep_a = 0, 0, []

        while True:
            if episode % 200 == 0:
                env.render()

            a, v = ppo.evaluate_state(s)

            # Update ppo
            if t == BATCH:  # or (terminal and t < BATCH):
                # Normalise rewards
                rewards = np.array(buffer_r)
                rolling_r.update(rewards)
                rewards = np.clip(rewards / rolling_r.std, -10, 10)

                v_final = [v * (1 - terminal)]  # vf = 0 if terminal, otherwise use the predicted vf
                values = np.array(buffer_v + v_final)
                terminals = np.array(buffer_terminal + [terminal])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(buffer_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                bs, ba, br, badv = np.reshape(buffer_s, (t,) + ppo.s_dim), np.vstack(buffer_a), \
                                   np.vstack(returns), np.vstack(advantage)

                graph_summary = ppo.train(bs, ba, br, badv)
                buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
                t = 0

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(v)
            buffer_terminal.append(terminal)
            ep_a.append(a)

            a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, _ = env.step(a)
            buffer_r.append(r)

            ep_r += r
            ep_t += 1
            t += 1

            if terminal:
                print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
                rew_list.append(ep_r)
                if episode % 100 == 0:
                    print("Mean reward of the past 100 episodes: ", str(np.mean(rew_list[-100:])))
                    mean_100.append(np.mean(rew_list[-100:]))
                    f = open('results.txt', 'a')
                    f.write('\n' + str(np.mean(rew_list[-100:])))
                    f.close()

                    # End of episode summary
                worker_summary = tf.Summary()
                worker_summary.value.add(tag="Perf/Reward", simple_value=ep_r)
                worker_summary.value.add(tag="Perf/Length", simple_value=ep_t)
                ppo.writer.add_summary(worker_summary, episode)
                try:
                    ppo.writer.add_summary(graph_summary, episode)
                except NameError:
                    pass
                ppo.writer.flush()

                # Save the model
                if episode % 1000 == 0 and episode > 0:
                    path = ppo.save_model(SUMMARY_DIR, episode)

                    print('Saved model at episode', episode, 'in', path)

                break
