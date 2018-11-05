from keras.layers.core import Layer
from keras import backend as K
import keras


def squash_activation(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + 1e-7)
    return scale * vectors


class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = keras.initializers.random_uniform(-1, 1)
        self.bias_initializer = keras.initializers.Zeros()
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.W = self.add_weight(shape=[input_shape[1], self.num_capsule, input_shape[2], self.dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.b = self.add_weight(shape=[input_shape[1], self.num_capsule],
                                 initializer=self.bias_initializer,
                                 name='b')
        self.c = self.add_weight(shape=[input_shape[1], self.num_capsule],
                                 initializer=self.bias_initializer,
                                 name='c')
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 2)
        inputs_tiled = K.repeat_elements(inputs_expand, self.num_capsule, axis=2)
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 2]), inputs_tiled)
        input_shape = K.shape(inputs_hat)
        b = self.b
        b = K.expand_dims(b, axis=0)
        c = self.c
        c = K.expand_dims(c, axis=0)
        assert self.num_routing > 0
        for i in range(self.num_routing):
            c = K.softmax(b)
            c = K.expand_dims(c, axis=-1)
            c = K.repeat_elements(c, rep=self.dim_capsule, axis=-1)
            S = K.sum(c * inputs_hat, axis=1)
            V = squash_activation(S)
            if i != self.num_routing-1:
                V_expanded = K.expand_dims(V, axis=1)
                V_expanded = K.tile(V_expanded, [1, input_shape[1], 1, 1])
                b = b + K.sum(inputs_hat * V_expanded, axis=-1)
        return V

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        base_config = super(CapsuleLayer, self).get_config()
        base_config['num_capsule'] = self.num_capsule
        base_config['num_routing'] = self.num_routing
        base_config['dim_capsule'] = self.dim_capsule
        return base_config
