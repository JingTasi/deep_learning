import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import utils.utils as utils


class PCA:

    def __init__(self, x_train, sess, dim_z=2,
                 lr=1e-3, epoch=100, batch_size=100, report_period=1000,
                 np_seed=1, tf_seed=1,
                 initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
                 activation=tf.nn.relu,
                 save_path='result/model/model_1',
                 figure_save_dir='result/img'):
        self.x_train = x_train
        self.sess = sess
        self.dim_z = dim_z
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.report_period = report_period
        self.np_seed = np_seed
        self.tf_seed = tf_seed
        self.initializer = initializer
        self.activation = activation
        self.save_path = save_path
        self.figure_save_dir = figure_save_dir

        self.feature_dim = self.x_train.shape[1]
        self.save_dir = self.save_path.split('/')[0] + '/' + self.save_path.split('/')[1]

        self.x = None
        self.E_W, self.E_b = None, None
        self.z = None
        self.D_W, self.D_b = None, None
        self.x_recon = None
        self.cost = None
        self.opt = None
        self.saver = None

    def construct_graph(self):
        np.random.seed(self.np_seed)
        tf.set_random_seed(self.tf_seed)

        input_size = self.feature_dim

        self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        self.z, _, _, _ = self.layer(self.x,
                                     [input_size, self.dim_z], [self.dim_z],
                                     'E_W', 'E_b', 'z', 'prob_z',
                                     is_active=False)
        self.x_recon, _, _, _ = self.layer(self.z,
                                           [self.dim_z, input_size], [input_size],
                                           'D_W', 'D_b', 'x_recon', 'prob_x_recon',
                                           is_active=False)

        self.cost = tf.nn.l2_loss(self.x - self.x_recon)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def layer(self, x, w_shape, b_shape, w_name, b_name, logits_name, probs_name, is_active=True):
        w = tf.get_variable(w_name, shape=w_shape, initializer=self.initializer)
        b = tf.get_variable(b_name, initializer=tf.zeros(b_shape))
        logits = tf.matmul(x, w) + b
        if is_active:
            logits = self.activation(logits)
        probs = tf.nn.sigmoid(logits)
        return tf.identity(logits, name=logits_name), tf.identity(probs, name=probs_name), w, b

    def layers(self, x, w1, b1, w2, b2, w3, b3):
        h1 = self.activation(tf.matmul(x, w1) + b1)
        h2 = self.activation(tf.matmul(h1, w2) + b2)
        logits = self.activation(tf.matmul(h2, w3) + b3)
        probs = tf.nn.sigmoid(logits)
        return logits, probs

    def layer_w_and_b(self, w_shape, b_shape, w_name, b_name):
        w = tf.get_variable(w_name, shape=w_shape, initializer=self.initializer)
        b = tf.get_variable(b_name, initializer=tf.zeros(b_shape))
        return w, b

    def plot_16_generated(self, figure_index=0):
        if not os.path.exists(self.figure_save_dir):
            os.makedirs(self.figure_save_dir)

        z = np.random.normal(0., 1., size=(16, self.dim_z))
        feed_dict = {self.z: z}
        images = self.sess.run(self.x_recon, feed_dict=feed_dict)

        n = images.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)

        fig = utils.plot_16_images_2d_and_return(images, img_shape=img_shape)
        fig.savefig(self.figure_save_dir + '/{}.png'.format(figure_index), bbox_inches='tight')
        plt.close(fig)

    def plot_16_loading_vectors(self):
        z = np.zeros(shape=(16, self.dim_z))
        for i in range(16):
            if i < self.dim_z:
                z[i, i] = 1

        feed_dict = {self.z: z}
        images = self.sess.run(self.x_recon, feed_dict=feed_dict)

        n = images.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)

        fig = utils.plot_16_images_2d_and_return(images, img_shape=img_shape)
        plt.show(fig)

    def plot_16_original_and_recon(self, img_original):
        n = img_original.shape[1]
        m = int(np.sqrt(n))
        img_shape = (m, m)

        fig = utils.plot_16_images_2d_and_return(img_original, img_shape=img_shape)
        plt.show(fig)

        img_recon = self.recon(img_original)
        fig = utils.plot_16_images_2d_and_return(img_recon, img_shape=img_shape)
        plt.show(fig)

    def recon(self, x):
        feed_dict = {self.x: x}
        return self.sess.run(self.x_recon, feed_dict=feed_dict)

    def restore(self):
        self.saver = tf.train.import_meta_graph(self.save_path + '.meta', clear_devices=True)
        self.saver.restore(sess=self.sess, save_path=self.save_path)

        self.x = self.sess.graph.get_tensor_by_name('x:0')
        self.x_recon = self.sess.graph.get_tensor_by_name('x_recon:0')

    def save(self):
        self.saver = tf.train.Saver()
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver.save(sess=self.sess, save_path=self.save_path)

    @staticmethod
    def shuffle_data(*args):
        idx = np.arange(args[0].shape[0])
        np.random.shuffle(idx)
        list_to_return = []
        for arg in args:
            list_to_return.append(arg[idx])
        return list_to_return

    def train(self):
        self.construct_graph()
        tf.global_variables_initializer().run()

        gradient_step = 0
        for epoch_num in range(self.epoch):
            x = self.shuffle_data(self.x_train)
            x = x[0]
            for i in range(self.x_train.shape[0] // self.batch_size):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x: x_batch}

                gradient_step += 1
                if gradient_step % self.report_period == 0:
                    loss, _ = self.sess.run([self.cost, self.opt], feed_dict=feed_dict)
                    print('gradient_step : ', gradient_step)
                    print('loss :           ', loss)
                    print()
                else:
                    self.sess.run(self.opt, feed_dict=feed_dict)

        self.save()
