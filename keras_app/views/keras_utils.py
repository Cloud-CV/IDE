import math

def shape_input(i_s,config):
	return config['batch_input_shape']

def shape_convolution(i_s,config):
	pad = config['border_mode'] if 'border_mode' in config else 'valid'
	o_c = config['nb_filter'] if 'nb_filter' in config else i_s[1]
	stride_h = config['subsample'][0]
	stride_w = config['subsample'][1]
	if pad == 'same':
		o_h = int(math.ceil(float(i_s[2])/stride_h))
		o_w = int(math.ceil(float(i_s[3])/stride_w))
	else:
		o_h = int(math.ceil(float(i_s[2] - config['nb_row'] + 1)/stride_h))
		o_w = int(math.ceil(float(i_s[3] - config['nb_col'] + 1)/stride_w))
	return [i_s[0],o_c,o_h,o_w]

def shape_pool(i_s,config):
	pad = config['border_mode'] if 'border_mode' in config else 'valid'
	o_c = config['nb_filter'] if 'nb_filter' in config else i_s[1]
	stride_h = config['strides'][0]
	stride_w = config['strides'][1]
	if pad == 'same':
		o_h = int(math.ceil(float(i_s[2])/stride_h))
		o_w = int(math.ceil(float(i_s[3])/stride_w))
	else:
		o_h = int(math.ceil(float(i_s[2] - config['pool_size'][0] + 1)/stride_h))
		o_w = int(math.ceil(float(i_s[3] - config['pool_size'][1] + 1)/stride_w))
	return [i_s[0],o_c,o_h,o_w]

def shape_dense(i_s,config):
	return [i_s[0],config['output_dim']]

def shape_concat(i_s_list,config):
	o_s = [i for i in i_s_list[0]]
	o_s[1] = 0
	for s in i_s_list:
		o_s[1] += s[1]
	return o_s

def shape_identity(i_s,config):
	return i_s

def shape_flatten(i_s,config):
	return [i_s[0],i_s[1]*i_s[2]*i_s[3]]

def shape_zero_pad(i_s,config):
	pad_h = config['padding'][0]
	pad_w = config['padding'][1]
	return [i_s[0],i_s[1],i_s[2] + 2*pad_h,i_s[3] + 2*pad_w]

def shape_pool_helper(i_s,config):
	n_h = config['n_h']
	n_w = config['n_w']
	return [i_s[0],i_s[1],i_s[2]-n_h,i_s[3]-n_w]

def shape_error(i_s,config):
	raise Exception('Cannot determine dimensions of data layer')

shape_map = {
	'InputLayer' : shape_input,
	'Convolution2D' : shape_convolution,
	'Dense' : shape_dense,
	'MaxPooling2D' : shape_pool,
	'AveragePooling2D' : shape_pool,
	'Merge' : shape_concat,
	'Activation' : shape_identity,
	'Dropout' : shape_identity,
	'LRN' : shape_identity,
	'ZeroPadding2D': shape_zero_pad,
	'Flatten': shape_flatten,
	'PoolHelper': shape_pool_helper
}

# find keras layer shapes from model.json
def find_keras_shapes(data):
	net_length = len(data['config']['layers'])

	shape = []
	for i in range(net_length):
		shape.append([0,0,0,0])

	name_map = {}
	for i in range(net_length):
		name_map[data['config']['layers'][i]['name']] = i

	for i in range(net_length):
		layer = data['config']['layers'][i]
		layer_type = layer['class_name']
		config = layer['config']
		i_s = []
		if len(layer['inbound_nodes']):
			for k in layer['inbound_nodes'][0]:
				i_s.append(shape[name_map[k[0]]]['o'])
		if len(i_s) == 1:
			i_s = i_s[0]
		shape[i] = {'o':shape_map[layer_type](i_s,config),'i':i_s}

	shape_dict = {}
	for i in range(net_length):
		shape_dict['l' + str(i)] = shape[i]
	return shape_dict

def print_keras_shapes(data):
	shape = find_keras_shapes(data)
	for i in range(len(shape)):
		print '{0: <30}'.format(data['config']['layers'][i]['name']),'\t',shape['l' + str(i)]['o']
