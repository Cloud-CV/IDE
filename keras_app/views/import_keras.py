from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from .keras2json import keras_to_json

def import_keras_json(request):
    if request.method == 'POST':
        try:
            data_file = request.FILES['file']
        except Exception:
            return JsonResponse({'result': 'error', 'error': 'No model.json file found'})

        data = json.load(data_file)
        net = keras_to_json(data)
        net_name = data['config']['name']
        return JsonResponse({'result': 'success', 'net': net, 'net_name':net_name })
