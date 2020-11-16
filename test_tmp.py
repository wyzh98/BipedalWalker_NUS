import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  
a1 = tf.get_variable(name='a1', shape=[2,3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
a3 = tf.get_variable(name='a3', shape=[2,3], initializer=tf.ones_initializer())
log_sigma = tf.get_variable(name="pi_sigma", shape=4, initializer=tf.zeros_initializer())
dist = tf.distributions.Normal(loc=[0., 10.], scale=[2., 1.])
sample = dist.sample(10)
mode = dist.mode()
sigma = dist.stddev()
b = tf.reduce_mean(sigma)
# a = dist.prob([2., 10.])

 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(a1))
    # print(sess.run(a2))
    # print(sess.run(a3))
    # print(sess.run(log_sigma))
    print(sess.run(b))
    plt.show()
