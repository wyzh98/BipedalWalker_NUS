import tensorflow as tf
import numpy as np
import os
from time import time


class PPO:
    def __init__(self, env, summary_dir='./', gpu=False):

        self.LR = 1e-4
        self.MINIBATCH = 64
        self.EPOCHS = 8
        self.EPSILON = 0.2
        self.EPS_LEN = 100000
        # GPU setup
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': gpu})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # Placeholders
        self.sess = tf.Session(config=config)
        self.s_dim, self.a_dim = env.observation_space.shape, env.action_space.shape[0]
        self.a_bound = (env.action_space.high - env.action_space.low) / 2
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.state = tf.placeholder(tf.float32, [None, self.s_dim[0]], 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # Dateset with experiennce replay
        self.dataset = tf.data.Dataset.from_tensor_slices({'state': self.state, 'actions': self.actions,
                                                           'rewards': self.rewards, 'advantage': self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(self.MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(self.EPOCHS)
        self.data_iter = self.dataset.make_initializable_iterator()
        batch = self.data_iter.get_next()
        # Call ppo net
        pi_old, pi_old_params = self.build_anet(batch['state'], 'oldpi')
        pi, pi_params = self.build_anet(batch['state'], 'pi')
        pi_eval, _ = self.build_anet(self.state, 'pi', reuse=True)

        vf_old, vf_old_params = self.build_cnet(batch['state'], 'oldvf')
        self.vf, vf_params = self.build_cnet(batch['state'], 'vf')
        self.vf_eval, _ = self.build_cnet(self.state, 'vf', reuse=True)

        self.sample_action = tf.squeeze(pi_eval.sample(1), axis=0)
        self.eval_action = pi_eval.mode()
        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver()
        # Loss functions and training
        epsilon_decay = tf.train.polynomial_decay(self.EPSILON, self.global_step, self.EPS_LEN, 0.1, power=1)
        ratio = tf.maximum(pi.prob(batch['actions']), 1e-6) / tf.maximum(pi_old.prob(batch['actions']), 1e-6)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = batch['advantage'] * ratio
        surr2 = batch['advantage'] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        loss_pg = - 2.0 * tf.reduce_mean(tf.minimum(surr1, surr2))
        loss_vf = 0.5 * tf.reduce_mean(tf.square(batch['rewards'] - self.vf))
        loss_entropy = - 0.01 * tf.reduce_mean(pi.entropy())
        loss = loss_pg + loss_vf + loss_entropy
        opt = tf.train.AdamOptimizer(self.LR)
        self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)

        self.pi_new_params = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
        self.vf_new_params = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]
        self.sess.run(tf.global_variables_initializer())

        # Tensorboard
        self.writer = tf.summary.FileWriter(summary_dir)
        tf.summary.scalar('Loss/Policy', loss_pg)
        tf.summary.scalar('Loss/Value', loss_vf)
        tf.summary.scalar('Loss/Entropy', loss_entropy)
        tf.summary.scalar('Loss/Total', loss)
        tf.summary.scalar('Var/Epsilon', epsilon_decay)
        tf.summary.scalar('Var/Policy Mode', tf.reduce_mean(pi.mode()))
        tf.summary.scalar('Var/Policy Sigma', tf.reduce_mean(pi.stddev()))
        tf.summary.scalar('Var/Value', tf.reduce_mean(self.vf))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # AC net
    def build_anet(self, state_in, name, reuse=False):
        reg = tf.contrib.layers.l2_regularizer(1e-3)
        with tf.variable_scope(name, reuse=reuse):
            layer_a1 = tf.layers.dense(state_in, 512, tf.nn.relu, kernel_regularizer=reg)
            layer_a2 = tf.layers.dense(layer_a1, 256, tf.nn.relu, kernel_regularizer=reg)
            layer_a3 = tf.layers.dense(layer_a2, 256, tf.nn.relu, kernel_regularizer=reg)
            mu = tf.layers.dense(layer_a2+layer_a3, self.a_dim, tf.nn.tanh, kernel_regularizer=reg)
            sigma = tf.layers.dense(layer_a2, self.a_dim, tf.nn.softplus, kernel_regularizer=reg)
            # sigma = tf.get_variable(name='pi_sigma', shape=self.a_dim, initializer=tf.constant_initializer(0.5))
            sigma = tf.clip_by_value(sigma, 0.0, 1.0)
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def build_cnet(self, state_in, name, reuse=False):
        reg = tf.contrib.layers.l2_regularizer(1e-3)
        with tf.variable_scope(name, reuse=reuse):
            layer_c1 = tf.layers.dense(state_in, 256, tf.nn.relu, kernel_regularizer=reg)
            layer_c2 = tf.layers.dense(layer_c1, 256, tf.nn.relu, kernel_regularizer=reg)
            layer_c3 = tf.layers.dense(layer_c2, 256, tf.nn.relu, kernel_regularizer=reg)
            vf = tf.layers.dense(layer_c2+layer_c3, 1, kernel_regularizer=reg)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params

    # Update the network
    def train(self, s, a, r, adv):
        start = time()
        self.sess.run([self.pi_new_params, self.vf_new_params, self.data_iter.initializer],
                      feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})
        while True:
            try:
                summary, step, _ = self.sess.run([self.summarise, self.global_step, self.train_op])
            except tf.errors.OutOfRangeError:
                break
        print('\rTrained in %.3fs. Global step %i' % (time() - start, step+1))
        return summary

    def choose_state(self, state, training=True):
        if training:
            a, v = self.sess.run([self.sample_action, self.vf_eval], {self.state: [state]})
        else:
            a, v = self.sess.run([self.eval_action, self.vf_eval], {self.state: [state]})
        return a[0], np.squeeze(v)

    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, 'model.ckpt'), global_step=step)
        return save_path

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)
        print('Model restored from', model_path)
