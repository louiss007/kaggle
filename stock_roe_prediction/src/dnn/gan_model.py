"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-3-6 下午6:11
# @FileName: gan_model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class gan_model:

    def __init__(self, model_para, out_para, task_type=None):
        """

        :param model_para:
        :param out_para:
        :param task_type:
        """
        # nn_model.__init__(self, model_para, out_para, task_type)
        self.height = 28
        self.width = 28
        self.channels = 1
        self.dropout = model_para.get('dropout')
        self.learning_rate = model_para.get('learning_rate')
        self.epoch = model_para.get('epoch')
        # self.n_step = model_para.get('n_step')
        self.train_sample_size = model_para.get('train_sample_size')
        self.test_sample_size = model_para.get('test_sample_size')
        self.batch_size = model_para.get('batch_size')
        self.n_step = self.train_sample_size // self.batch_size + 1
        self.num_classes = model_para.get('num_classes')
        self.noise_dim = model_para.get('noise_dim')
        self.image_dim = model_para.get('num_input')
        self.gen_hidden_dim = model_para.get('g_hidden')
        self.disc_hidden_dim = model_para.get('g_hidden')
        self.global_step = tf.Variable(0, trainable=False)
        self.model_path = '{mp}/nn/gan/gan'.format(mp=out_para.get('model_path'))
        self.init_net()
        self.g_loss, self.d_loss, self.g_train, self.d_train = self.build_model()

    def init_net(self):
        # Network Inputs
        self.g_x = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='input_noise')
        self.d_x = tf.placeholder(tf.float32, shape=[None, self.image_dim], name='disc_input')
        self.weights = {
            'gen_hidden1': tf.Variable(self.glorot_init([self.noise_dim, self.gen_hidden_dim])),
            'gen_out': tf.Variable(self.glorot_init([self.gen_hidden_dim, self.image_dim])),
            'disc_hidden1': tf.Variable(self.glorot_init([self.image_dim, self.disc_hidden_dim])),
            'disc_out': tf.Variable(self.glorot_init([self.disc_hidden_dim, 1]))
        }
        self.biases = {
            'gen_hidden1': tf.Variable(tf.zeros([self.gen_hidden_dim])),
            'gen_out': tf.Variable(tf.zeros([self.image_dim])),
            'disc_hidden1': tf.Variable(tf.zeros([self.disc_hidden_dim])),
            'disc_out': tf.Variable(tf.zeros([1]))
        }

    def neural_network(self):
        pass

    def build_model(self):
        # Build Generator Network
        g_sample = self.generator(self.g_x)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        d_real = self.discriminator(self.d_x)
        d_fake = self.discriminator(g_sample)

        # Build Loss
        g_loss = -tf.reduce_mean(tf.log(d_fake), keep_dims=True)
        d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake), keep_dims=True)

        # Build Optimizers
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        g_vars = [self.weights['gen_hidden1'], self.weights['gen_out'],
                    self.biases['gen_hidden1'], self.biases['gen_out']]
        # Discriminator Network Variables
        d_vars = [self.weights['disc_hidden1'], self.weights['disc_out'],
                     self.biases['disc_hidden1'], self.biases['disc_out']]

        # Create training operations
        # TODO add global_step=self.global_step in minimize, leading the following error:
        # tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible
        # shapes: [96, 1]
        # vs.[128, 1]
        # [[{{node gradients_1 / add_6_grad / BroadcastGradientArgs}}]]
        g_train = g_optimizer.minimize(g_loss, var_list=g_vars)
        d_train = d_optimizer.minimize(d_loss, var_list=d_vars)
        self.g_sample = g_sample
        return g_loss, d_loss, g_train, d_train

    def fit(self, sess, feed_dict):
        """
        learn model, 训练模型
        :param sess:
        :param batch_x: 特征
        :param batch_y: 标签
        :return:
        """
        g_loss, d_loss, _, _, step = sess.run(
            [self.g_loss, self.d_loss, self.g_train, self.d_train, self.global_step], feed_dict=feed_dict)
        return g_loss, d_loss, step

    # def predict(self, sess, x, y):
    #     """
    #     模型预测
    #     :param sess:
    #     :param x: 特征
    #     :param y: y is None
    #     :return: y_hat
    #     """
    #     result = sess.run([self.out], feed_dict={
    #         self.d_x: x,
    #         self.g_x: y
    #     })
    #     return result

    def eval(self):
        pass

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + '.ckpt', global_step=self.global_step)

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    def parse_tfrecord(self, tfrecord):
        example = tf.parse_single_example(tfrecord, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'num1': tf.FixedLenFeature([], tf.float32),
            'num2': tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(example['image'], tf.float32)
        label = tf.decode_raw(example['label'], tf.float32)
        # num1 = example['num1']
        # num2 = example['num2']
        image = tf.reshape(image, shape=[self.height, self.width, self.channels])
        image = tf.reshape(image, shape=[self.image_dim])
        label = tf.reshape(label, shape=[self.num_classes])
        return image, label

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

    def make_one_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(256)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_one_batch_for_g(self, batch_x):
        # Generate noise to feed to the generator
        # z = tf.random.uniform([self.batch_size, self.noise_dim], -1., 1.)
        z = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])
        feed_dict = {
            self.d_x: batch_x,
            self.g_x: z
        }
        return feed_dict

    def show_fake_image(self, sess):
        # Generate images from noise, using the generator network.
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[4, self.noise_dim])
            g = sess.run([self.g_sample], feed_dict={self.g_x: z})
            g = tf.reshape(g, shape=(4, 28, 28, 1))
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(4):
                # Generate image from noise. Extend to 3 channels for matplot figure.
                img = tf.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                 shape=(28, 28, 3))
                a[j][i].imshow(img)

        f.show()
        plt.draw()
        plt.waitforbuttonpress()

    @staticmethod
    def check_shape(t):
        print('==========================')
        print(t.op.name, ' ', t.get_shape().as_list())
