from keras.layers.core import Layer
from keras import backend as K


class MaskCapsule(Layer):
    def __init__(self, **kwargs):
        super(MaskCapsule, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCapsule, self).build(input_shape)

    def call(self, inputs):
        if type(inputs) == list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        masked = K.batch_flatten(inputs*K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1]*input_shape[0][2]])
        else:
            return tuple([None, input_shape[1]*input_shape[2]])

    def get_config(self):
        return super(MaskCapsule, self).get_config()
