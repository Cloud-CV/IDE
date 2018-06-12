import random
import string
import sys

from caffe_app.models import Network, SharedWith
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from yaml import safe_load
from django.shortcuts import render
from django.contrib.auth.models import User


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def save_to_db(request):
    if request.method == 'POST':
        net = request.POST.get('net')
        net_name = request.POST.get('net_name')
        if net_name == '':
            net_name = 'Net'
        try:
            # randomId = datetime.now().strftime('%Y%m%d%H%M%S')+randomword(5)
            # author and shared with fix users until a login feature is added
            # author of model will be received in request object
            # user with whom model is shared will be received in request object
            model_id = int(request.POST.get('networkId'))

            model = Network.objects.get(id=model_id)
            model.name = net_name
            model.network = net
            model.author = User.objects.get(id=1)
            model.save()

            sharedWith = SharedWith(network=model, user=User.objects.get(id=2),
                                    access_privilege='E')
            sharedWith.save()
            return JsonResponse({'result': 'success', 'id': model_id})
        except:
            return JsonResponse({'result': 'error', 'error': str(sys.exc_info()[1])})


@csrf_exempt
def load_from_db(request):
    if request.method == 'POST':
        if 'proto_id' in request.POST:
            try:
                user_id = int(request.POST['user_id'])
                model = Network.objects.get(pk=request.POST['proto_id'])
                net = safe_load(model.network)

                sharedModels = SharedWith.objects.filter(network=model, user=user_id)
                # authorizing the user for access to model
                if not sharedModels and user_id != model.author.id:
                    return JsonResponse({'result': 'error',
                                         'error': 'Permission denied for access to model'})
            except Exception:
                return JsonResponse({'result': 'error',
                                     'error': 'No network file found'})
            return JsonResponse({'result': 'success', 'net': net, 'net_name': model.name})

    if request.method == 'GET':
        return index(request)
