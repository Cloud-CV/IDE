import math


def Input(layer):
    params = {}
    shape = layer['config']['batch_input_shape']
    params['dim'] = str([1, shape[3], shape[1], shape[2]])[1:-1]
    return jsonLayer('Input', params, layer)


def Convolution(layer):
    params = {}
    params['kernel_w'] = layer['config']['kernel_size'][0]
    params['kernel_h'] = layer['config']['kernel_size'][1]
    params['stride_w'] = layer['config']['strides'][0]
    params['stride_h'] = layer['config']['strides'][1]
    params['pad_w'], params['pad_h'] = get_padding(params['kernel_w'], params['kernel_h'],
                                                   params['stride_w'], params['stride_h'],
                                                   layer['config']['padding'].lower())
    params['weight_filler'] = layer['config']['kernel_initializer']['class_name']
    params['bias_filler'] = layer['config']['bias_initializer']['class_name']
    params['num_output'] = layer['config']['filters']
    return jsonLayer('Convolution', params, layer)


def Pooling(layer, shape=None):
    params = {}
    poolMap = {
        'MaxPooling2D': 0,
        'AveragePooling2D': 1,
        'GlobalAveragePooling2D': 1
    }
    # For GAP
    if (shape is not None):
        params['kernel_w'] = shape[2]
        params['kernel_h'] = shape[1]
        params['stride_w'] = shape[2]
        params['stride_h'] = shape[1]
        params['pad_w'], params['pad_h'] = get_padding(params['kernel_w'], params['kernel_h'],
                                                       params['stride_w'], params['stride_h'],
                                                       'valid')
    else: 
        params['kernel_w'] = layer['config']['pool_size'][0]
        params['kernel_h'] = layer['config']['pool_size'][1]
        params['stride_w'] = layer['config']['strides'][0]
        params['stride_h'] = layer['config']['strides'][1]
        params['pad_w'], params['pad_h'] = get_padding(params['kernel_w'], params['kernel_h'],
                                                       params['stride_w'], params['stride_h'],
                                                       layer['config']['padding'].lower())
    params['pool'] = poolMap[layer['class_name']]
    return jsonLayer('Pooling', params, layer)


def Flatten(layer):
    params = {}
    return jsonLayer('Flatten', params, layer)


def Concat(layer):
    return jsonLayer('Concat', {}, layer)


def Dense(layer):
    params = {}
    params['weight_filler'] = layer['config']['kernel_initializer']['class_name']
    params['bias_filler'] = layer['config']['bias_initializer']['class_name']
    params['num_output'] = layer['config']['units']
    return jsonLayer('InnerProduct', params, layer)


def Activation(layer):
    activationMap = {
        'softmax': 'Softmax',
        'elu': 'ELU',
        'relu': 'ReLU',
        'tanh': 'TanH',
        'sigmoid': 'Sigmoid'
    }
    if (layer['class_name'] == 'Activation'):
        return jsonLayer(activationMap[layer['config']['activation']], {}, layer)
    else:
        tempLayer = {}
        tempLayer['inbound_nodes'] = [[[layer['name']+layer['class_name']]]]
        return jsonLayer(activationMap[layer['config']['activation']], {}, tempLayer)


def Eltwise(layer):
    eltwiseMap = {
        'Add': 1,
        'Multiply': 0,
        'Maximum': 2
    }
    params = {'operation': eltwiseMap[layer['class_name']]}
    return jsonLayer('Eltwise', params, layer)


def Scale(layer):
    tempLayer = {}
    params = {'bias_term': layer['config']['center']}
    tempLayer['inbound_nodes'] = [[[layer['name']+layer['class_name']]]]
    return jsonLayer('Scale', params, tempLayer)


def Padding(layer):
    pad = layer['config']['padding']
    params = {'pad_h': pad[0][0], 'pad_w': pad[1][0]}
    return jsonLayer('Pad', params, layer)


def BatchNorm(layer):
    params = {}
    params['eps'] = layer['config']['epsilon']
    params['moving_average_fraction'] = layer['config']['momentum']
    return jsonLayer('BatchNorm', params, layer)


def get_padding(k_w, k_h, s_w, s_h, pad_type):
    if (pad_type == 'valid'):
        return [0, 0]
    else:
        if (s_w != 1 or s_h != 1):
            raise Exception('Cannot calculate padding for stride '+str((s_w, s_h)))
        else:
            return (math.ceil((k_w-1)/2), math.ceil((k_h-1)/2))


def jsonLayer(type, params, layer):
    input = []
    if (len(layer['inbound_nodes'])):
        for node in layer['inbound_nodes'][0]:
            input.append(node[0])
    layer = {
                'info': {
                    'type': type,
                    'phase': None
                },
                'connection': {
                    'input': input,
                    'output': []
                },
                'params': params
            }
    return layer
