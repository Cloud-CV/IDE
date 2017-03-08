from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ide.utils.jsonToPrototxt import jsonToPrototxt
from ide.utils.json_utils import get_inputs
import os
import random, string
import sys
import tensorflow as tf
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR+'/media/')
sys.path.insert(0, BASE_DIR+'/tensorflow_app/caffe-tensorflow/')
from convert import convert

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def exportToTensorflow(request):
    if request.method == 'POST':
        net = yaml.safe_load(request.POST.get('net'))
        net_name = request.POST.get('net_name')
        if net_name == '':
            net_name = 'Net'

        # rename input layers to 'data'
        inputs = get_inputs(net)
        for i in inputs:
            net[i]['props']['name'] = 'data'

        prototxt,input_dim = jsonToPrototxt(net,net_name)
        randomId=datetime.now().strftime('%Y%m%d%H%M%S')+randomword(5)
        with open(BASE_DIR+'/media/'+randomId+'.prototxt', 'w') as f:
            f.write(prototxt)

        convert(BASE_DIR+'/media/'+randomId+'.prototxt', None, None, BASE_DIR+'/media/'+randomId+'.py', 'test')

        # NCHW to NHWC data format
        input_caffe = input_dim
        input_tensorflow = []
        for i in [0,2,3,1]:
            input_tensorflow.append(input_caffe[i])

        # converting generated caffe-tensorflow code to graphdef
        try:
            net = __import__ (str(randomId))
            images = tf.placeholder(tf.float32, input_tensorflow)
            # the name of the first layer should be 'data' !
            net = getattr(net, net_name)({'data': images})
            graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
            with open(BASE_DIR+'/media/'+randomId+'.pbtxt', 'w') as f: f.write(str(graph_def))
        except AssertionError:
            return JsonResponse({'result': 'error', 'error': 'Cannot convert to GraphDef'})
        except AttributeError:
            return JsonResponse({'result': 'error', 'error': 'GraphDef not supported'})

        return JsonResponse({'result': 'success','id': randomId, 'name': randomId+'.pbtxt', 'url': '/media/'+randomId+'.pbtxt'})


