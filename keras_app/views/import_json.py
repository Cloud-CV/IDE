import json
import yaml

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from layers import *


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
        model = model['config']
    else:
        model = model['config']['layers']

    layer_map = {
        'InputLayer': Input,
        'Conv2D': Convolution,
        'relu': Activation,
        'softmax': Activation,
        'MaxPooling2D': Pooling,
        'Flatten': Flatten,
        'Dense': Dense
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
            else:
                net[layer['name']] = layer_map[layer['class_name']](layer)
                name = layer['name']
            if (layer['inbound_nodes']):
                for node in layer['inbound_nodes'][0]:
                    net[node[0]]['connection']['output'].append(name)
    return JsonResponse({'result': 'success', 'net': net, 'net_name': ''})
