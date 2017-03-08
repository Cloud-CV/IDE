import math
import json
from keras.layers.advanced_activations import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.layers import *
from keras.models import Model
from extra_layers import *
from ide.utils.json_utils import *

def check_for_same_padding(input_shape,output_shape,stride_h,stride_w,kernel_h,kernel_w,pad_h,pad_w):
	o_h = int(math.ceil(float(input_shape[2])/stride_h))
	o_w = int(math.ceil(float(input_shape[3])/stride_w))
	if output_shape[2] == o_h and output_shape[3] == o_w:
		# extra check for pooling layers
		p_h = (output_shape[2]-1)*stride_h + kernel_h - input_shape[2]
		p_w = (output_shape[3]-1)*stride_w + kernel_w - input_shape[3]
		if p_h%2 == 0 and p_w%2 == 0 and pad_h == p_h/2 and pad_w == p_w/2:
			return True
	return False

def check_for_valid_padding(input_shape,output_shape,stride_h,stride_w,kernel_h,kernel_w):
	o_h = int(math.ceil(float(input_shape[2] - kernel_h + 1)/stride_h))
	o_w = int(math.ceil(float(input_shape[3] - kernel_w + 1)/stride_w))
	if output_shape[2] == o_h and output_shape[3] == o_w:
		return True
	return False

def json_to_keras(net):

    k_layers = {}
    process_order = find_process_order(net)
    shape = find_caffe_shapes(net)

    for id in process_order:
        k_layers[id] = None

    for id in process_order:
        layer_type = net[id]['info']['type']
        name = net[id]['name']
        params = net[id]['params']
        input_shape = shape[id]['i']
        output_shape = shape[id]['o']
        inputs = net[id]['connection']['input']
        input_k_layers = [k_layers[input_id] for input_id in inputs]

        if layer_type == 'Input':
            k_layers[id] = Input(shape=tuple(output_shape[1:]), name=name)

        elif layer_type == 'Data':
            raise Exception('Cannot determine dimensions of data layer')

        elif layer_type == 'Convolution':
            pad_h = params['pad_h'] if 'pad_h' in params else 0
            pad_w = params['pad_w'] if 'pad_w' in params else 0
            stride_h = params['stride_h'] if 'stride_h' in params else 1
            stride_w = params['stride_w'] if 'stride_w' in params else 1
            kernel_h = params['kernel_h']
            kernel_w = params['kernel_w']
            num_output = params['num_output']
            if check_for_same_padding(input_shape,output_shape,stride_h,stride_w,kernel_h,kernel_w,pad_h,pad_w):
                k_layers[id] = Convolution2D(num_output, kernel_h, kernel_w, border_mode='same', subsample=(stride_h, stride_w), name=name)(input_k_layers)
            else:
                if pad_h + pad_w > 0:
                    input_k_layers = ZeroPadding2D(padding=(pad_h, pad_w), name=name + '_zeropadding')(input_k_layers)
                k_layers[id] = Convolution2D(num_output, kernel_h, kernel_w, border_mode='valid', subsample=(stride_h, stride_w), name=name)(input_k_layers)

        elif layer_type == 'InnerProduct':
            num_output = params['num_output']
            if len(input_k_layers[0]._keras_shape[1:]) > 1:
                input_k_layers = Flatten(name=name + '_flatten')(input_k_layers)
            k_layers[id] = Dense(num_output, name=name)(input_k_layers)

        elif layer_type == 'Pooling':
            pad_h = params['pad_h'] if 'pad_h' in params else 0
            pad_w = params['pad_w'] if 'pad_w' in params else 0
            stride_h = params['stride_h'] if 'stride_h' in params else 1
            stride_w = params['stride_w'] if 'stride_w' in params else 1
            kernel_h = params['kernel_h']
            kernel_w = params['kernel_w']
            # MAX pooling
            if params['pool'] == 0:
                if check_for_same_padding(input_shape,output_shape,stride_h,stride_w,kernel_h,kernel_w,pad_h,pad_w):
                    k_layers[id] = MaxPooling2D(pool_size=(kernel_h, kernel_w), border_mode='same', strides=(stride_h, stride_w), name=name)(input_k_layers)
                else:
                    o_h = output_shape[2]
                    o_w = output_shape[3]
                    i_h = input_shape[2]
                    i_w = input_shape[3]
                    extra_pad_h = ((o_h - 1)*stride_h + kernel_h) - (i_h + 2*pad_h)
                    extra_pad_w = ((o_w - 1)*stride_w + kernel_w) - (i_w + 2*pad_w)
                    total_pad_h = pad_h + extra_pad_h
                    total_pad_w = pad_w + extra_pad_w
                    if total_pad_h + total_pad_w > 0:
                        input_k_layers = ZeroPadding2D(padding=(total_pad_h, total_pad_w), name=name + '_zeropadding')(input_k_layers)
                        input_k_layers = PoolHelper(n_h=extra_pad_h, n_w=extra_pad_w, name=name + '_poolhelper')(input_k_layers)
                    k_layers[id] = MaxPooling2D(pool_size=(kernel_h, kernel_w), border_mode='valid', strides=(stride_h, stride_w), name=name)(input_k_layers)
            # AVE pooling
            elif params['pool'] == 1:
                if check_for_same_padding(input_shape,output_shape,stride_h,stride_w,kernel_h,kernel_w,pad_h,pad_w):
                    k_layers[id] = AveragePooling2D(pool_size=(kernel_h, kernel_w), border_mode='same', strides=(stride_h, stride_w), name=name)(input_k_layers)
                else:
                    o_h = output_shape[2]
                    o_w = output_shape[3]
                    i_h = input_shape[2]
                    i_w = input_shape[3]
                    extra_pad_h = ((o_h - 1)*stride_h + kernel_h) - (i_h + 2*pad_h)
                    extra_pad_w = ((o_w - 1)*stride_w + kernel_w) - (i_w + 2*pad_w)
                    total_pad_h = pad_h + extra_pad_h
                    total_pad_w = pad_w + extra_pad_w
                    if total_pad_h + total_pad_w > 0:
                        input_k_layers = ZeroPadding2D(padding=(total_pad_h, total_pad_w), name=name + '_zeropadding')(input_k_layers)
                        input_k_layers = PoolHelper(n_h=extra_pad_h, n_w=extra_pad_w)(input_k_layers)
                    k_layers[id] = AveragePooling2D(pool_size=(kernel_h, kernel_w), border_mode='valid', strides=(stride_h, stride_w), name=name)(input_k_layers)

        elif layer_type == 'Concat':
            k_layers[id] = merge(input_k_layers, mode='concat', concat_axis=1, name=name)

        elif layer_type == 'ReLU':
            k_layers[id] = Activation('relu', name=name)(input_k_layers)

        elif layer_type == 'Dropout':
            # dropout ratio!
            k_layers[id] = Dropout(0.4, name=name)(input_k_layers)

        elif layer_type == 'LRN':
            k_layers[id] = LRN(name=name)(input_k_layers)

        elif layer_type == 'Softmax':
            k_layers[id] = Activation('softmax', name=name)(input_k_layers)

        elif layer_type == 'SoftmaxWithLoss':
            k_layers[id] = Activation('softmax', name=name)(input_k_layers)

    inputs = get_inputs(net)
    outputs = get_outputs(net)
    input_l = [k_layers[id] for id in inputs]
    output_l = [k_layers[id] for id in outputs]
    model = Model(input=input_l, output=output_l)
    return model
