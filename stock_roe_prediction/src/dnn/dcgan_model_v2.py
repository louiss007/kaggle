"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-3-12 下午8:25
# @FileName: dcgan_model_v2.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from dnn.gan_model import gan_model


class dcgan_model(gan_model):

    def __init__(self, model_para, out_para, task_type=None):
        gan_model.__init__(self, model_para, out_para, task_type)
        self.global_step = tf.Variable(0, trainable=False)
        self.model_path = '{mp}/nn/dcgan/dcgan'.format(mp=out_para.get('model_path'))
        self.init_net()
        self.g_loss, self.d_loss, self.g_train, self.d_train = self.build_model()

    def init_net(self):
        # Network Inputs
        self._generator = Generator()
        self._discriminator = Discriminator()
        self.g_x = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='input_noise')
        self.d_x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name='disc_input')

    def build_model(self):
        g_sample = self._generator(self.g_x, training=True)
        fake_logits = self._discriminator(g_sample, training=True)
        real_logits = self._discriminator(self.d_x, training=True)
        d_loss_fake = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self.batch_size], dtype=tf.int32),
                logits=fake_logits
            )
        )
        d_loss_real = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self.batch_size], dtype=tf.int32),
                logits=real_logits
            )
        )
        g_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self.batch_size], dtype=tf.int32),
                logits=fake_logits
            )
        )

        d_loss = d_loss_fake + d_loss_real

        # Build Optimizers
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Create training operations
        g_train_op = g_optimizer.minimize(g_loss, var_list=self._generator.variables, global_step=self.global_step)
        d_train_op = d_optimizer.minimize(d_loss, var_list=self._discriminator.variables, global_step=self.global_step)
        return g_loss, d_loss, g_train_op, d_train_op

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
        # image = tf.reshape(image, shape=[self.image_dim])
        label = tf.reshape(label, shape=[self.num_classes])
        return image, label


class Generator(object):

    def __init__(self):
        # self._channels = channels
        # self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        x = tf.convert_to_tensor(inputs)
        # with tf.variable_scope('generator', reuse=self._reuse):
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        # self._reuse = True
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return x


class Discriminator(object):

    def __init__(self):
        # self._channels = channels
        # self._init_conv_size = init_conv_size
        self._reuse = False

    def __call__(self, inputs, training):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        conv_x = x
        # with tf.variable_scope('discriminator', reuse=self._reuse):
        x = tf.layers.conv2d(conv_x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
        # self._reuse = True
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return x
