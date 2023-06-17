import tensorflow as tf

class BasePlasticity(tf.keras.layers.Layer):
    def __init__(self, units, activation='linear', learning_rate=0.001, **kwargs):
        super(BasePlasticity, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.learning_rate = learning_rate
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        return self.activation(output)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def get_config(self):
        config = super(BasePlasticity, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'learning_rate': self.learning_rate
        })
        return config

class HebbianPlasticity(BasePlasticity):
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = tf.matmul(inputs, tf.transpose(post_activation)) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class HebbianModifiedPlasticity(BasePlasticity):
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        norm = tf.norm(self.kernel)
        delta_weight = tf.matmul(inputs, tf.transpose(post_activation)) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        self.kernel = tf.clip_by_norm(self.kernel, norm)  # Limita a norma dos pesos
        return post_activation

class AntiHebbianPlasticity(BasePlasticity):
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = -tf.matmul(inputs, tf.transpose(post_activation)) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class OjaPlasticity(BasePlasticity):
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = tf.matmul(inputs, tf.transpose(post_activation)) - \
            tf.square(post_activation) * tf.reduce_sum(tf.square(self.kernel), axis=0)
        delta_weight *= self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class BCMPlasticity(BasePlasticity):
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = tf.matmul(inputs * post_activation, tf.transpose(post_activation)) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class STDPPlasticity(BasePlasticity):
    def __init__(self, units, activation='linear', learning_rate=0.001, pre_tau=20, post_tau=20, **kwargs):
        super(STDPPlasticity, self).__init__(units, activation, learning_rate, **kwargs)
        self.pre_tau = pre_tau
        self.post_tau = post_tau
    
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = tf.matmul(tf.exp(-tf.abs(inputs) / self.pre_tau) * post_activation,
                                 tf.transpose(tf.exp(tf.minimum(inputs, 0.0)) / self.post_tau)) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class eSTDPPlasticity(BasePlasticity):
    def __init__(self, units, activation='linear', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20, post_tau_pos=20, post_tau_neg=20, **kwargs):
        super(eSTDPPlasticity, self).__init__(units, activation, learning_rate, **kwargs)
        self.pre_tau_pos = pre_tau_pos
        self.pre_tau_neg = pre_tau_neg
        self.post_tau_pos = post_tau_pos
        self.post_tau_neg = post_tau_neg
    
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = (tf.matmul(tf.exp(-tf.abs(inputs) / self.pre_tau_neg) * post_activation,
                                  tf.transpose(tf.exp(tf.minimum(inputs, 0.0)) / self.post_tau_neg)) +
                        tf.matmul(tf.exp(-tf.abs(inputs) / self.pre_tau_pos) * post_activation,
                                  tf.transpose(tf.exp(tf.maximum(inputs, 0.0)) / self.post_tau_pos))) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation

class iSTDPPlasticity(BasePlasticity):
    def __init__(self, units, activation='linear', learning_rate=0.001, pre_tau=20, post_tau_pos=20, post_tau_neg=20, **kwargs):
        super(iSTDPPlasticity, self).__init__(units, activation, learning_rate, **kwargs)
        self.pre_tau = pre_tau
        self.post_tau_pos = post_tau_pos
        self.post_tau_neg = post_tau_neg
    
    def call(self, inputs):
        pre_activation = tf.matmul(inputs, self.kernel)
        post_activation = self.activation(pre_activation)
        delta_weight = (tf.matmul(tf.exp(-tf.abs(inputs) / self.pre_tau) * post_activation,
                                  tf.transpose(tf.exp(tf.maximum(inputs, 0.0)) / self.post_tau_pos)) -
                        tf.matmul(tf.exp(-tf.abs(inputs) / self.pre_tau) * post_activation,
                                  tf.transpose(tf.exp(tf.minimum(inputs, 0.0)) / self.post_tau_neg))) * self.learning_rate
        self.kernel.assign_add(delta_weight)
        return post_activation