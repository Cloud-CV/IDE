from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import yaml
from datetime import datetime
import random
import string
import os

from ide.utils.shapes import get_shapes
# from layers_export import Input, Convolution, Deconvolution, Pooling, Dense, Dropout, Embed,\
#    Recurrent, BatchNorm, Activation, LeakyReLU, PReLU, Scale, Flatten, Reshape, Concat, Eltwise,\
#    Padding
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


@csrf_exempt
def exportJson(request):
    if request.method == 'POST':
        net = yaml.safe_load(request.POST.get('net'))
        net_name = request.POST.get('net_name')
        if net_name == '':
            net_name = 'Net'
        net = get_shapes(net)
