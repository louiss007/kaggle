"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-3-6 下午6:11
# @FileName: gan.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf


class gan_model():

    def __init__(self):
        pass

    def init_net(self):
        # Network Inputs
        g_x = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
        d_x = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')
        self.weights = {
            'gen_hidden1': tf.Variable(self.glorot_init([noise_dim, gen_hidden_dim])),
            'gen_out': tf.Variable(self.glorot_init([gen_hidden_dim, image_dim])),
            'disc_hidden1': tf.Variable(self.glorot_init([image_dim, disc_hidden_dim])),
            'disc_out': tf.Variable(self.glorot_init([disc_hidden_dim, 1])),
        }
        self.biases = {
            'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
            'gen_out': tf.Variable(tf.zeros([image_dim])),
            'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
            'disc_out': tf.Variable(tf.zeros([1])),
        }

    def neural_network(self):
        pass

    def glorot_init(self, shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    # Generator
    def generator(self, x):
        hidden_layer = tf.matmul(x, self.weights['gen_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['gen_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['gen_out'])
        out_layer = tf.add(out_layer, self.biases['gen_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    # Discriminator
    def discriminator(self, x):
        hidden_layer = tf.matmul(x, self.weights['disc_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['disc_hidden1'])
        hidden_layer = tf.nn.relu(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['disc_out'])
        out_layer = tf.add(out_layer, self.biases['disc_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer
