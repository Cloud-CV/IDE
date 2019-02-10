from keras.layers.core import Layer
from keras import backend as K


class Squash(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Squash, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Squash, self).build(input_shape)

    def call(self, inputs):
        s_squared_norm = K.sum(K.square(inputs), self.axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + 1e-7)
        return scale * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(Squash, self).get_config()
        base_config['axis'] = self.axis
        return base_config
