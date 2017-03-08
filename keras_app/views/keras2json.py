from ide.utils.json_utils import find_process_order
import json
from .keras_utils import find_keras_shapes


'''
with open('custom_net_keras.json') as data_file:
    data = json.load(data_file)
'''

layer_map = {
    'Convolution2D': 'Convolution',
    'LRN': 'LRN',
    'Merge': 'Concat',
    'InputLayer': 'Input',
    'AveragePooling2D': 'Pooling',
    'MaxPooling2D': 'Pooling',
    'Dropout': 'Dropout',
    'PoolHelper': 'keras_pool_helper',
    'Dense': 'keras_dense',
    'Activation': 'keras_activation',
    'ZeroPadding2D': 'keras_zero_pad',
    'Flatten': 'keras_flatten'
}

def keras_to_json(data):
    phase = None #!
    net_length = len(data['config']['layers'])
    name_map = {}
    for i in range(net_length):
        name_map[data['config']['layers'][i]['name']] = 'l' + str(i)

    # create the json network
    net = {}
    for i in range(net_length):
        id = 'l' + str(i)
        layer = data['config']['layers'][i]
        layer_type = layer['class_name']
        if layer_type in layer_map:
            caffe_layer_type = layer_map[layer_type]
        else:
            raise NotImplementedError
        name = layer['name']
        config = layer['config']
        params = {}

        if layer_type == 'InputLayer':
            input_shape = str([k if k else 0 for k in config['batch_input_shape']])[1:-1] # 0 for None?
            params['dim'] = input_shape

        elif layer_type == 'Convolution2D':
            if 'nb_row' in config:
                params['kernel_h'] = config['nb_row']
            else:
                raise Exception('Missing parameters in "' + name + '" layer')
            if 'nb_col' in config:
                params['kernel_w'] = config['nb_col']
            else:
                raise Exception('Missing parameters in "' + name + '" layer')
            if 'subsample' in config:
                params['stride_h'] = config['subsample'][0]
                params['stride_w'] = config['subsample'][1]
            else:
                raise Exception('Missing parameters in "' + name + '" layer')
            params['keras_pad'] = config['border_mode']
            if 'nb_filter' in config:
                params['num_output'] = config['nb_filter']


        elif layer_type == 'AveragePooling2D':
            if 'pool_size' in config:
                params['kernel_h'] = config['pool_size'][0]
                params['kernel_w'] = config['pool_size'][1]
            else:
                raise Exception('Missing parameters in "' + name + '" layer in keras json')
            if 'strides' in config:
                params['stride_h'] = config['strides'][0]
                params['stride_w'] = config['strides'][1]
            else:
                raise Exception('Missing parameters in "' + name + '" layer in keras json')
            params['keras_pad'] = config['border_mode']
            params['pool'] = 'AVE'

        elif layer_type == 'MaxPooling2D':
            if 'pool_size' in config:
                params['kernel_h'] = config['pool_size'][0]
                params['kernel_w'] = config['pool_size'][1]
            else:
                raise Exception('Missing parameters in "' + name + '" layer in keras json')
            if 'strides' in config:
                params['stride_h'] = config['strides'][0]
                params['stride_w'] = config['strides'][1]
            else:
                raise Exception('Missing parameters in "' + name + '" layer in keras json')
            params['keras_pad'] = config['border_mode']
            params['pool'] = 'MAX'

        elif layer_type == 'PoolHelper':
            params['n_h'] = config['n_h']
            params['n_w'] = config['n_w']

        elif layer_type == 'Dense':
            params['output_dim'] = config['output_dim']

        elif layer_type == 'Activation':
            if config['activation'] == 'relu':
                caffe_layer_type = 'ReLU'
            elif config['activation'] == 'softmax':
                caffe_layer_type = 'Softmax'

        elif layer_type == 'ZeroPadding2D':
            params['pad_h'] = config['padding'][0]
            params['pad_w'] = config['padding'][1]

        elif layer_type == 'Dropout':
            pass

        elif layer_type == 'LRN':
            pass

        elif layer_type == 'Merge':
            pass

        elif layer_type == 'Flatten':
            pass

        inputs = []
        if len(layer['inbound_nodes']):
            for k in layer['inbound_nodes'][0]:
                inputs.append(name_map[k[0]])

        json_layer = {
            'name': name,
            'info': {
                'type': caffe_layer_type,
                'phase': phase
            },
            'connection': {
                'input': inputs,
                'output': []
            },
            'params': params
        }
        net[id] = json_layer

        for k in inputs:
            if not id in net[k]['connection']['output']:
                net[k]['connection']['output'].append(id)

    shape = find_keras_shapes(data)
    new_id = net_length

    process_order = find_process_order(net)
    for id in process_order:
        if id in net:
            # convert Flatten + Dense into InnerProduct
            if net[id]['info']['type'] == 'keras_flatten' :
                if net[net[id]['connection']['output'][0]]['info']['type'] == 'keras_dense':
                    flatten_layer = net[id]
                    dense_layer_id = net[id]['connection']['output'][0]
                    dense_layer = net[dense_layer_id]

                    # create new inner product layer
                    params = {}
                    params['num_output'] = dense_layer['params']['output_dim']
                    inner_product_layer = {
                        'name': dense_layer['name'],
                        'info': {
                            'type': 'InnerProduct',
                            'phase': phase
                        },
                        'connection': {
                            'input': [i for i in flatten_layer['connection']['input']],
                            'output': [i for i in dense_layer['connection']['output']]
                        },
                        'params': params
                    }
                    new_layer_id = 'l' + str(new_id)
                    name_map[dense_layer['name']] = new_id
                    net[new_layer_id] = inner_product_layer
                    new_id +=1

                    # delete previous connections and make new
                    for i in flatten_layer['connection']['input']:
                        net[i]['connection']['output'].remove(id)
                        net[i]['connection']['output'].append(new_layer_id)
                    for i in dense_layer['connection']['output']:
                        net[i]['connection']['input'].remove(dense_layer_id)
                        net[i]['connection']['input'].append(new_layer_id)

                    # delete Flatten and Dense layer
                    del net[id]
                    del net[dense_layer_id]
                else:
                    raise Exception('Invalid use of Flatten layer')

            # convert ZeroPadding + Convolution2D into Convolution
            elif net[id]['info']['type'] == 'keras_zero_pad' and net[net[id]['connection']['output'][0]]['info']['type'] == 'Convolution':
                zero_pad_layer = net[id]
                conv_layer_id = net[id]['connection']['output'][0]
                conv_layer = net[conv_layer_id]
                if conv_layer['params']['keras_pad'] == 'valid':
                    conv_layer['params']['pad_h'] = zero_pad_layer['params']['pad_h']
                    conv_layer['params']['pad_w'] = zero_pad_layer['params']['pad_w']
                    del conv_layer['params']['keras_pad']
                    conv_layer['connection']['input'] = [i for i in zero_pad_layer['connection']['input']]
                    for i in conv_layer['connection']['input']:
                        net[i]['connection']['output'].remove(id)
                        net[i]['connection']['output'].append(conv_layer_id)
                    del net[id]
                else:
                    raise Exception('Convolution with \'same\' padding after ZeroPadding layer')

            # convert ZeroPadding + Pooling into Pooling
            elif net[id]['info']['type'] == 'keras_zero_pad' and net[net[id]['connection']['output'][0]]['info']['type'] == 'Pooling':
                zero_pad_layer = net[id]
                pool_layer_id = net[id]['connection']['output'][0]
                pool_layer = net[pool_layer_id]
                input_shape = shape[id]['i']
                output_shape = shape[id]['o']
                if pool_layer['params']['keras_pad'] == 'valid':
                    pad_h = zero_pad_layer['params']['pad_h']
                    pad_w = zero_pad_layer['params']['pad_w']
                    p_h = (output_shape[2]-1)*pool_layer['params']['stride_h'] + pool_layer['params']['kernel_h'] - input_shape[2]
                    p_w = (output_shape[3]-1)*pool_layer['params']['stride_w'] + pool_layer['params']['kernel_w'] - input_shape[3]
                    if p_h%2 == 0 and p_w%2 == 0 and p_h/2 == pad_h and p_w/2 == pad_w:
                        pool_layer['params']['pad_h'] = pad_h
                        pool_layer['params']['pad_w'] = pad_w
                        del pool_layer['params']['keras_pad']
                        pool_layer['connection']['input'] = [i for i in zero_pad_layer['connection']['input']]
                        for i in pool_layer['connection']['input']:
                            net[i]['connection']['output'].remove(id)
                            net[i]['connection']['output'].append(pool_layer_id)
                        del net[id]
                    else:
                        raise Exception('Cannot convert due to differences in pooling op b/w caffe and keras. Consider using PoolHelper')
                else:
                    raise Exception('Pooling with \'same\' padding after ZeroPadding layer')

            # convert ZeroPadding + PoolHelper + Pooling into Pooling
            elif net[id]['info']['type'] == 'keras_zero_pad':
                pool_helper_id = net[id]['connection']['output'][0]
                if net[pool_helper_id]['info']['type'] == 'keras_pool_helper':
                    pool_id = net[pool_helper_id]['connection']['output'][0]
                    if net[pool_id]['info']['type'] == 'Pooling':
                        zero_pad_layer = net[id]
                        pool_helper_layer = net[pool_helper_id]
                        pool_layer = net[pool_id]
                        zero_pad_h = zero_pad_layer['params']['pad_h']
                        zero_pad_w = zero_pad_layer['params']['pad_w']
                        n_h = pool_helper_layer['params']['n_h']
                        n_w = pool_helper_layer['params']['n_w']
                        kernel_h = pool_layer['params']['kernel_h']
                        kernel_w = pool_layer['params']['kernel_w']
                        stride_h = pool_layer['params']['stride_h']
                        stride_w = pool_layer['params']['stride_w']
                        input_shape = shape[id]['i']
                        output_shape = shape[pool_id]['o']
                        if pool_layer['params']['keras_pad'] == 'valid':
                            if output_shape[2] == float(input_shape[2] + 2*zero_pad_h - n_h - kernel_h)/stride_h +1 \
                                and output_shape[3] == float(input_shape[3] + 2*zero_pad_w - n_w - kernel_w)/stride_w +1 :
                                pool_layer['params']['pad_h'] = zero_pad_h - n_h
                                pool_layer['params']['pad_w'] = zero_pad_w - n_w
                                del pool_layer['params']['keras_pad']
                                pool_layer['connection']['input'] = [i for i in zero_pad_layer['connection']['input']]
                                for i in pool_layer['connection']['input']:
                                    net[i]['connection']['output'].remove(id)
                                    net[i]['connection']['output'].append(pool_id)
                                del net[id]
                                del net[pool_helper_id]
                            else:
                                raise Exception('Cannot convert due to differences in pooling op b/w caffe and keras.')
                        else:
                            raise Exception('Pooling with \'same\' padding after ZeroPadding layer')

    # at this point, there should be no ZeroPadding, PoolHelper, Dense or Flatten layers in the network
    for id in net.keys():
        layer_type = net[id]['info']['type']
        if layer_type == 'keras_zero_pad' or layer_type == 'keras_pool_helper' or layer_type == 'keras_dense' or layer_type == 'keras_flatten':
            raise Exception('Network not supported')

    # calculate padding values for remaining conv and pool layers
    for id in net.keys():
        layer_type = net[id]['info']['type']
        if layer_type == 'Convolution' and 'keras_pad' in net[id]['params']:
            layer = net[id]
            input_shape = shape[id]['i']
            output_shape = shape[id]['o']
            keras_pad = layer['params']['keras_pad']
            stride_h = layer['params']['stride_h']
            stride_w = layer['params']['stride_w']
            kernel_h = layer['params']['kernel_h']
            kernel_w = layer['params']['kernel_w']
            if keras_pad == 'valid':
                layer['params']['pad_h'] = 0
                layer['params']['pad_w'] = 0
            elif keras_pad == 'same':
                p_h = (output_shape[2]-1)*stride_h + kernel_h - input_shape[2]
                p_w = (output_shape[3]-1)*stride_w + kernel_w - input_shape[3]
                if p_h%2 == 0 and p_w%2 == 0:
                    layer['params']['pad_h'] = p_h/2
                    layer['params']['pad_w'] = p_w/2
                else:
                    raise Exception('Cannot convert due to differences in pooling schemes b/w caffe and keras.')
            del layer['params']['keras_pad']

        elif layer_type == 'Pooling' and 'keras_pad' in net[id]['params']:
            layer = net[id]
            input_shape = shape[id]['i']
            output_shape = shape[id]['o']
            keras_pad = layer['params']['keras_pad']
            stride_h = layer['params']['stride_h']
            stride_w = layer['params']['stride_w']
            kernel_h = layer['params']['kernel_h']
            kernel_w = layer['params']['kernel_w']
            if keras_pad == 'valid':
                if (output_shape[2]-1)*stride_h + kernel_h - input_shape[2] == 0 and (output_shape[3]-1)*stride_w + kernel_w - input_shape[3] == 0 :
                    layer['params']['pad_h'] = 0
                    layer['params']['pad_w'] = 0
                else :
                    raise Exception('Cannot convert due to differences in pooling op b/w caffe and keras.')
            elif keras_pad == 'same':
                p_h = (output_shape[2]-1)*stride_h + kernel_h - input_shape[2]
                p_w = (output_shape[3]-1)*stride_w + kernel_w - input_shape[3]
                if p_h%2 == 0 and p_w%2 == 0:
                    layer['params']['pad_h'] = p_h/2
                    layer['params']['pad_w'] = p_w/2
                else:
                    raise Exception('Cannot convert due to differences in pooling schemes b/w caffe and keras.')
            del layer['params']['keras_pad']
    '''
    temp = find_process_order(net)
    temp1 = []
    for i in temp:
        temp1.append(net[i])
    print json.dumps(temp1)
    '''
    #print json.dumps(net)
    return net
