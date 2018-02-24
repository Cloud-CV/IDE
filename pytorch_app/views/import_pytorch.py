from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
import json
import requests
from datetime import datetime
import random
import string


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


@csrf_exempt
def import_model(request):
    loadFromText = False
    if request.method == 'POST':
        if ('file' in request.FILES):
            f = request.FILES['file']
        elif 'sample_id' in request.POST:
            try:
                f = open(os.path.join(settings.BASE_DIR,
                                      'example', 'pytorch',
                                      request.POST['sample_id'] + '.pt'), 'r')
            except Exception:
                return JsonResponse({'result': 'error',
                                     'error': 'No model file found'})
        elif 'config' in request.POST:
            loadFromText = True
        try:
            if loadFromText is True:
                model = request.POST['config']
            else:
                model = f
        except Exception:
            return JsonResponse({'result': 'error', 'error': 'Invalid File'})

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))

    randomId = datetime.now().strftime('%Y%m%d%H%M%S') + randomword(5)
    with open(BASE_DIR + "/media/" + randomId, "wb") as f:
        f.write(model.read())
    os.chdir(BASE_DIR + '/pytorch_app/views/')
    os.system("KERAS_BACKEND=tensorflow python pytorch2json.py -input_file " + BASE_DIR +
              "/media/" + randomId + " -output_file " + BASE_DIR + "/media/" + randomId + ".json")
    with open(BASE_DIR + "/media/" + randomId + ".json", "r") as f:
        json_string = f.read()
    print json_string[:100]
    r = requests.post("http://localhost:8000/keras/import",
                      data={"config": json_string})

    return JsonResponse(json.loads(r.content))
