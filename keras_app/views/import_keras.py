from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from keras2json import keras_to_json
import json

@csrf_exempt
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
