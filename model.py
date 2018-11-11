import numpy as np
import tensorflow as tf

from _layers import one_residual


class TensorflowModel:
    def __init__(self, observation_shape, action_shape, graph_location, seed=1234):
        # model definition
        self._observation = None
        self._action = None
        self._computation_graph = None
        self._optimization_op = None

        self.tf_session = tf.InteractiveSession()

        # restoring
        self.tf_checkpoint = None
        self.tf_saver = None

        self.seed = seed

        self._initialize(observation_shape, action_shape, graph_location)

    def predict(self, state):
        action = self.tf_session.run(self._computation_graph, feed_dict={
            self._observation: [state],
        })
        return np.squeeze(action)

    def train(self, observations, actions):
        _, loss = self.tf_session.run([self._optimization_op, self._loss], feed_dict={
            self._observation: observations,
            self._action: actions
        })
        return loss

    def commit(self):
        self.tf_saver.save(self.tf_session, self.tf_checkpoint)

    def computation_graph(self):
        #model = one_residual(self._preprocessed_state, seed=self.seed)

        output = tf.layers.conv2d(self._preprocessed_state, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.conv2d(output, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

        output = tf.layers.conv2d(output, filters=128, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.conv2d(output, filters=128, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

        output = tf.layers.conv2d(output, filters=256, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.max_pooling2d(output, pool_size=2, strides=4)

        output = tf.layers.conv2d(output, filters=512, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.keras.initializers.he_normal(seed=self.seed),
                              kernel_regularizer=tf.keras.regularizers.l2(1e-04))
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)

        output = tf.layers.max_pooling2d(output, pool_size=2, strides=4)

        output = tf.layers.flatten(output)

        output = tf.layers.dense(output, units=256, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        output = tf.layers.dense(output, units=128, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))

        output = tf.layers.dense(output, self._action.shape[1])

        return output

    def _optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=0.0001)

    def _loss_function(self):
        return tf.losses.mean_squared_error(self._action, self._computation_graph)

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)
            self.tf_session.run(tf.global_variables_initializer())

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self._observation)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize

    def _create(self, input_shape, output_shape):
        self._observation = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
        self._pre_process()

        self._computation_graph = self.computation_graph()
        self._loss = self._loss_function()
        self._optimization_op = self._optimizer().minimize(self._loss)

    def _storing(self, location):
        self.tf_saver = tf.train.Saver()

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            self.tf_checkpoint = location

    def close(self):
        self.tf_session.close()
