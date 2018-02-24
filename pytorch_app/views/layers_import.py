import numpy as np
import keras.layers


def convert_dense(node, node_name, input_name, output_name, layers):

    print('Converting dense ...')
    W = \
        node.next_functions[2][0].next_functions[0][0].variable.data.numpy()
    output_channels, input_channels = W.shape
    W = np.transpose(W)
    weights = [W]

    if node.next_functions[0][0]:
        bias = node.next_functions[0][0].variable.data.numpy()
        has_bias = True
        weights = [W, bias]
    else:
        has_bias = False

    dense = keras.layers.Dense(
        output_channels,
        weights=weights, use_bias=has_bias, name=output_name
    )
    layers[output_name] = dense(layers[input_name])


def convert_reshape(node, node_name, input_name, output_name, layers):

    print('Converting reshape ...')
    target_shape = node.new_sizes
    reshape = keras.layers.Reshape(target_shape[1:], name=output_name)
    layers[output_name] = reshape(layers[input_name])


def convert_convolution(node, node_name, input_name, output_name, layers):

    print('Converting convolution ...')
    weights_var = node.next_functions[1][0].variable

    if node.next_functions[2][0]:
        biases = node.next_functions[2][0].variable.data.numpy()
        has_bias = True
    else:
        biases = None
        has_bias = False

    if len(weights_var.data.numpy().shape) == 4:
        if node.transposed:
            W = weights_var.data.numpy().transpose(2, 3, 1, 0)
            height, width, n_filters, channels = W.shape
        else:
            W = weights_var.data.numpy().transpose(2, 3, 1, 0)
            height, width, channels, n_filters = W.shape

        assert node.output_padding == (0, 0)
        border_mode = 'valid'

        if node.padding[0] == node.padding[1] and node.padding[0] > 0:
            padding_name = output_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=node.padding,
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        if node.padding[0] != node.padding[1]:
            if node.padding[0] == height // 2 and\
               node.padding[1] == width // 2:
                border_mode = 'same'
            else:
                raise ValueError('Unsuported padding size for convolution')

        if node.dilation[0] != node.dilation[1]:
            raise ValueError('Unsuported dilation rate for convolution')
        dilation_rate = node.dilation[0]

        weights = None
        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        if node.transposed:

            conv = keras.layers.Conv2DTranspose(
                filters=n_filters,
                kernel_size=(height, width),
                strides=(node.stride[0], node.stride[1]),
                padding=border_mode,
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation_rate,
                name=output_name
            )
        else:
            conv = keras.layers.Conv2D(
                filters=n_filters,
                kernel_size=(height, width),
                strides=(node.stride[0], node.stride[1]),
                padding=border_mode,
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation_rate,
                name=output_name
            )
        layers[output_name] = conv(layers[input_name])
    else:
        if node.transposed:
            raise ValueError('Conv1d(transposed) is unsupported')
        else:
            W = weights_var.data.numpy().transpose(2, 1, 0)
            width, channels, n_filters = W.shape

        assert node.output_padding == (0, 0)

        if node.padding[0] != node.padding[1]:
            raise ValueError('Unsuported padding size for convolution')

        if node.dilation[0] != node.dilation[1]:
            raise ValueError('Unsuported dilation rate for convolution')
        dilation_rate = node.dilation[0]

        padding = node.padding[0]
        if padding > 0:
            padding_name = output_name + '_pad'
            padding_layer = keras.layers.ZeroPadding1D(
                padding=padding,
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        weights = None
        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        border_mode = 'valid'

        conv = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=(width),
            strides=(node.stride[0]),
            padding=border_mode,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation_rate,
            name=output_name
        )

        layers[output_name] = conv(layers[input_name])


def convert_batchnorm(node, node_name, input_name, output_name, layers):

    print('Converting batchnorm ...')

    weights = None
    biases = None
    if node.next_functions[1][0]:
        weights = node.next_functions[1][0].variable.data
        biases = node.next_functions[2][0].variable.data
        gamma = weights.numpy()
        beta = biases.numpy()

    mean = node.running_mean.numpy()
    variance = node.running_var.numpy()

    eps = node.eps
    momentum = node.momentum

    if weights is None:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=[mean, variance],
            name=output_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=[gamma, beta, mean, variance],
            name=output_name
        )
    layers[output_name] = bn(layers[input_name])


def convert_pooling(node, node_name, input_name, output_name, layers):

    print('Converting pooling ...')

    height, width = node.kernel_size
    stride_height, stride_width = node.stride

    if isinstance(node.padding, tuple) and node.padding[0] != node.padding[1]:
        raise ValueError('Unsuported padding size for pooling')

    if isinstance(node.padding, int):
        padding = node.padding
    else:
        padding = node.padding[0]

    border_mode = 'valid'

    if padding > 0:
        padding_name = output_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=node.padding,
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[input_name])
        input_name = padding_name

    if node_name.startswith('Max'):
        pooling = keras.layers.MaxPooling2D(
            pool_size=node.kernel_size,
            strides=(stride_height, stride_width),
            padding=border_mode,
            name=output_name
        )
    elif node_name.startswith('Avg'):
        pooling = keras.layers.AveragePooling2D(
            pool_size=node.kernel_size,
            strides=(stride_height, stride_width),
            padding=border_mode,
            name=output_name
        )
    else:
        raise ValueError('Unknown pooling type')

    layers[output_name] = pooling(layers[input_name])


def convert_threshold(node, node_name, input_name, output_name, layers):

    print('Converting RELU ...')
    relu = keras.layers.Activation('relu', name=output_name)
    layers[output_name] = relu(layers[input_name])


def convert_leakyrelu(node, node_name, input_name, output_name, layers):

    print('Converting LeakyRELU ...')
    leakyrelu = \
        keras.layers.LeakyReLU(alpha=node.additional_args[0], name=output_name)
    layers[output_name] = leakyrelu(layers[input_name])


def convert_prelu(node, node_name, input_name, output_name, layers):

    print('Converting PReLU ...')
    a = node.next_functions[1][0].variable.data.numpy()
    prelu = keras.layers.PReLU(name=output_name, weights=np.array([a]))
    layers[output_name] = prelu(layers[input_name])


def convert_selu(node, node_name, input_name, output_name, layers):

    print('Converting SELU ...')
    selu = keras.layers.Activation('selu', name=output_name)
    layers[output_name] = selu(layers[input_name])


def convert_tanh(node, node_name, input_name, output_name, layers):

    print('Converting tanh ...')
    tanh = keras.layers.Activation('tanh', name=output_name)
    layers[output_name] = tanh(layers[input_name])


def convert_softmax(node, node_name, input_name, output_name, layers):

    print('Converting softmax ...')
    softmax = keras.layers.Activation('softmax', name=output_name)
    layers[output_name] = softmax(layers[input_name])


def convert_softplus(node, node_name, input_name, output_name, layers):

    print('Converting softplus ...')
    softplus = keras.layers.Activation('softplus', name=output_name)
    layers[output_name] = softplus(layers[input_name])


def convert_softsign(node, node_name, input_name, output_name, layers):

    print('Converting softsign ...')
    softsign = keras.layers.Activation('softsign', name=output_name)
    layers[output_name] = softsign(layers[input_name])


def convert_sigmoid(node, node_name, input_name, output_name, layers):

    print('Converting sigmoid ...')
    sigmoid = keras.layers.Activation('sigmoid', name=output_name)
    layers[output_name] = sigmoid(layers[input_name])


def convert_dropout(node, node_name, input_name, output_name, layers):

    print('Converting dropout ...')

    dropout = keras.layers.Dropout(rate=node.p)
    layers[output_name] = dropout(layers[input_name])


def convert_elementwise_add(node, node_name, input_names, output_name, layers):

    print('Converting elementwise_add ...')
    model0 = layers[input_names[0]]
    model1 = layers[input_names[1]]

    add = keras.layers.Add(name=output_name)
    layers[output_name] = add([model0, model1])


def convert_elementwise_mul(node, node_name, input_names, output_name, layers):

    print('Converting elementwise_mul ...')
    model0 = layers[input_names[0]]
    model1 = layers[input_names[1]]

    mul = keras.layers.Multiply(name=output_name)
    layers[output_name] = mul([model0, model1])


def convert_elementwise_sub(node, node_name, input_names, output_name, layers):

    print('Converting elementwise_sub ...')
    model0 = layers[input_names[0]]
    model1 = layers[input_names[1]]

    sub = keras.layers.Subtract(name=output_name)
    layers[output_name] = sub([model0, model1])


def convert_concat(node, node_name, input_names, output_name, layers):

    print('Converting concat ...')
    concat_nodes = [layers[i] for i in input_names]
    cat = keras.layers.Concatenate(name=output_name, axis=node.dim)
    layers[output_name] = cat(concat_nodes)


AVAILABLE_CONVERTERS = {
    'Addmm': convert_dense,
    'ConvNd': convert_convolution,
    'BatchNorm': convert_batchnorm,
    'MaxPool2d': convert_pooling,
    'AvgPool2d': convert_pooling,
    'View': convert_reshape,
    'Threshold': convert_threshold,
    'LeakyReLU': convert_leakyrelu,
    'PReLU': convert_prelu,
    'SELU': convert_selu,
    'Tanh': convert_tanh,
    'Softmax': convert_softmax,
    'Softplus': convert_softplus,
    'Softsign': convert_softsign,
    'Sigmoid': convert_sigmoid,
    'Dropout': convert_dropout,
    'Add': convert_elementwise_add,
    'Mul': convert_elementwise_mul,
    'Sub': convert_elementwise_sub,
    'Concat': convert_concat,
}
