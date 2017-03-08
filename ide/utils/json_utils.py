import math

# find inputs of the network
def get_inputs(net):
	inputs = []
	for id in net:
		if(net[id]['info']['type'] == 'Data' or net[id]['info']['type'] == 'Input'):
			inputs.append(id)
	return inputs

# find outputs of the network
def get_outputs(net):
	outputs = []
	for id in net:
		if not len(net[id]['connection']['output']):
			outputs.append(id)
	return outputs

def find_process_order(net):
	stack = []
	layers_processed = {}
	process_order = []
	for id in net:
		layers_processed[id] = False

	def is_process_possible(id):
		inputs = net[id]['connection']['input']
		for input_id in inputs:
			if layers_processed[input_id] is False:
				return False
		return True

	# add input or data layers in stack
	stack.extend(get_inputs(net))
	while len(stack):
		i = len(stack) - 1
		while is_process_possible(stack[i]) is False:
			i = i - 1
		id = stack[i]
		stack.remove(stack[i])
		for output_id in net[id]['connection']['output']:
			if output_id not in stack:
				stack.append(output_id)
		layers_processed[id] = True
		process_order.append(id)

	return process_order

def shape_input(i_s,params):
    return params['dim']

def shape_convolution(i_s,params):
    pad_h = params['pad_h'] if 'pad_h' in params else 0
    pad_w = params['pad_w'] if 'pad_w' in params else 0
    stride_h = params['stride_h'] if 'stride_h' in params else 1
    stride_w = params['stride_w'] if 'stride_w' in params else 1
    o_h = int(math.floor(float(i_s[2] + 2*pad_h - params['kernel_h'])/stride_h) + 1)
    o_w = int(math.floor(float(i_s[3] + 2*pad_w - params['kernel_w'])/stride_w) + 1)
    o_c = params['num_output'] if params['num_output'] else i_s[1]
    return [i_s[0],o_c,o_h,o_w]

def shape_pool(i_s,params):
    pad_h = params['pad_h'] if 'pad_h' in params else 0
    pad_w = params['pad_w'] if 'pad_w' in params else 0
    stride_h = params['stride_h'] if 'stride_h' in params else 1
    stride_w = params['stride_w'] if 'stride_w' in params else 1
    o_h = int(math.ceil(float(i_s[2] + 2*pad_h - params['kernel_h'])/stride_h) + 1)
    o_w = int(math.ceil(float(i_s[3] + 2*pad_w - params['kernel_w'])/stride_w) + 1)
    return [i_s[0],i_s[1],o_h,o_w]

def shape_inner_product(i_s,params):
    return [i_s[0],params['num_output']]

def shape_concat(i_s_list,params):
    o_s = [i for i in i_s_list[0]]
    o_s[1] = 0
    for s in i_s_list:
        o_s[1] += s[1]
    return o_s

def shape_identity(i_s,params):
    return i_s

def shape_error(i_s,params):
    raise Exception('Cannot determine dimensions of data layer')

shape_map = {
	'Input' : shape_input,
	'Data' : shape_error,
	'Convolution' : shape_convolution,
	'InnerProduct' : shape_inner_product,
	'Pooling' : shape_pool,
	'Concat' : shape_concat,
	'ReLU' : shape_identity,
	'Accuracy' : shape_identity,
	'Dropout' : shape_identity,
	'LRN' : shape_identity,
	'Softmax' : shape_identity,
	'SoftmaxWithLoss' : shape_identity
}

def find_caffe_shapes(net):
	net_length = len(net)
	shape = {}

	process_order = find_process_order(net)

	for id in process_order:
		layer_type = net[id]['info']['type']
		input_length = len(net[id]['connection']['input'])
		if input_length == 1:
			i_s = shape[net[id]['connection']['input'][0]]['o']
		else:
			i_s = [shape[k]['o'] for k in net[id]['connection']['input']]
		shape[id] = { 'o':shape_map[layer_type](i_s,net[id]['params']), 'i':i_s }

	return shape

def print_json_shapes(net):
	shape = find_caffe_shapes(net)
	temp = find_process_order(net)
	print 'TOTAL NO OF LAYERS',len(temp)
	for id in temp:
		print '{0: <30}'.format(net[id]['name']),'\t',shape[id]['o']

def preprocess_json(net):
    for id in net.keys():
        layer_type = net[id]['info']['type']
        params = net[id]['params']
        net[id]['name'] = net[id]['props']['name']
        del net[id]['props']
        # remove unknown parameters
        for p in params.keys():
            if params[p] == '':
                del params[p]

        if layer_type == 'Input':
            params['dim'] = map(int,params['dim'].split(','))

        elif layer_type == 'Data':
            pass

        elif layer_type == 'Convolution':
            for p in params.keys():
                if p in ['kernel_h','kernel_w','stride_h','stride_w','pad_h','pad_w','num_output']:
                    params[p] = int(params[p])

        elif layer_type == 'Pooling':
            for p in params.keys():
                if p in ['kernel_h','kernel_w','stride_h','stride_w','pad_h','pad_w']:
                    params[p] = int(params[p])
            if 'pool' in params:
                if params['pool'] == 'MAX':
                    params['pool'] = 0
                elif params['pool'] == 'AVE':
                    params['pool'] = 1
                elif(params['pool'] == 'STOCHASTIC'):
                    params['pool'] = 2

        elif layer_type == 'ReLU':
            pass

        elif layer_type == 'InnerProduct':
            for p in params.keys():
                if p in ['num_output']:
                    params[p] = int(params[p])

        elif layer_type == 'SoftmaxWithLoss':
            pass

        elif layer_type == 'Accuracy':
            pass

        elif layer_type == 'Dropout':
            pass

        elif layer_type == 'LRN':
            pass

        elif layer_type == 'Concat':
            pass

        elif layer_type == 'Softmax':
            pass

    return net
