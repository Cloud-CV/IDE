from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ide.utils.json_utils import preprocess_json
from .json2keras import json_to_keras
import os
import random, string
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))

def export_keras_json(request):
    if request.method == 'POST':
        net = yaml.safe_load(request.POST.get('net'))
        net_name = request.POST.get('net_name')
        if net_name == '':
            net_name = 'Net'
        net = preprocess_json(net)
        model = json_to_keras(net)
        json_string = model.to_json()
        randomId=datetime.now().strftime('%Y%m%d%H%M%S')+randomword(5)
        with open(BASE_DIR+'/media/'+randomId+'.json', 'w') as f:
            f.write(json_string)
        return JsonResponse({'result': 'success','id': randomId, 'name': randomId+'.json', 'url': '/media/'+randomId+'.json'})

