import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from layers import Input, Convolution, Activation, Pooling, Dense, Flatten, Padding, BatchNorm,\
 Scale, Eltwise


@csrf_exempt
def importJson(request):
    if request.method == 'POST':
        try:
            f = request.FILES['file']
        except Exception:
            return JsonResponse({'result': 'error', 'error': 'No JSON model file found'})

        try:
            model = json.load(f)
        except Exception:
            return JsonResponse({'result': 'error', 'error': 'Invalid JSON'})

    if (model['class_name'] == 'Sequential'):
        net_name = ''
        model = model['config']
    else:
        net_name = model['config']['name']
        model = model['config']['layers']

    layer_map = {
        'InputLayer': Input,
        'Conv2D': Convolution,
        'relu': Activation,
        'softmax': Activation,
        'MaxPooling2D': Pooling,
        'AveragePooling2D': Pooling,
        'Flatten': Flatten,
        'Dense': Dense,
        'ZeroPadding2D': Padding,
        'BatchNormalization': BatchNorm,
        'Activation': Activation,
        'Add': Eltwise,

    }

    hasActivation = ['Conv2D', 'Dense']

    net = {}
    for layer in model:
        name = ''
        if (layer['class_name'] in layer_map):
            # This extra logic is to handle connections if the layer has an Activation
            if (layer['class_name'] in hasActivation and layer['config']['activation'] != 'linear'):
                net[layer['name']+layer['class_name']] = layer_map[layer['class_name']](layer)
                net[layer['name']] = layer_map[layer['config']['activation']](layer)
                net[layer['name']+layer['class_name']]['connection']['output'].append(layer['name'])
                name = layer['name']+layer['class_name']
            # To check if a Scale layer is required
            elif (layer['class_name'] == 'BatchNormalization' and (
                    layer['config']['center'] or layer['config']['scale'])):
                net[layer['name']+layer['class_name']] = layer_map[layer['class_name']](layer)
                net[layer['name']] = Scale(layer)
                net[layer['name']+layer['class_name']]['connection']['output'].append(layer['name'])
                name = layer['name']+layer['class_name']
            else:
                net[layer['name']] = layer_map[layer['class_name']](layer)
                name = layer['name']
            if (layer['inbound_nodes']):
                for node in layer['inbound_nodes'][0]:
                    net[node[0]]['connection']['output'].append(name)
    # collect names of all zeroPad layers
    zeroPad = []
    # Transfer parameters and connections from zero pad
    for node in net:
        if (net[node]['info']['type'] == 'Pad'):
            net[net[node]['connection']['output'][0]]['connection']['input'] = \
                net[node]['connection']['input']
            net[net[node]['connection']['output'][0]]['params']['pad_w'] = \
                net[node]['params']['pad_w']
            net[net[node]['connection']['output'][0]]['params']['pad_h'] = \
                net[node]['params']['pad_h']
            net[net[node]['connection']['input'][0]]['connection']['output'] = \
                net[node]['connection']['output']
            zeroPad.append(node)
        # Switching connection order to handle visualization
        elif (net[node]['info']['type'] == 'Eltwise'):
            net[node]['connection']['input'] = net[node]['connection']['input'][::-1]
    for node in zeroPad:
        net.pop(node, None)
    return JsonResponse({'result': 'success', 'net': net, 'net_name': net_name})
