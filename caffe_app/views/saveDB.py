from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from datetime import datetime
import random
import string
import sys
from caffe_app.models import ModelExport


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


@csrf_exempt
def saveToDB(request):
    if request.method == 'POST':
        net = request.POST.get('net')
        net_name = request.POST.get('net_name')
        if net_name == '':
            net_name = 'Net'
        try:
            randomId = datetime.now().strftime('%Y%m%d%H%M%S')+randomword(5)
            model = ModelExport(name=net_name, id=randomId, network=net)
            model.save()
            return JsonResponse({'result': 'success', 'id': randomId})
        except:
            return JsonResponse({'result': 'error', 'error': str(sys.exc_info()[1])})
