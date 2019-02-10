from keras.layers.core import Layer
from keras import backend as K


class Length(Layer):
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Length, self).build(input_shape)

    def call(self, input):
        return K.sqrt(K.sum(K.square(input), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        base_config = super(Length, self).get_config()
        return base_config
