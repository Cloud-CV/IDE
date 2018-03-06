import torch
from torch.autograd import Variable
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import urllib2
from urlparse import urlparse
from layers_import import extract_nodes

"""
layer_map is a dictionary mapping PyTorch operation
names with standard defined Caffe names
"""


layer_map = {
    'AddmmBackward': 'Dense',
    'MmBackward': 'Dense'
}


@csrf_exempt
def import_model(request):

    if request.method == 'POST':
        if ('file' in request.FILES) and \
           (request.FILES['file'].content_type == 'application/octet-stream' or
                request.FILES['file'].content_type == 'text/plain'):
            try:
                f = request.FILES['file']
                config = torch.load(f)
            except Exception:
                return JsonResponse({'result': 'error', 'error': 'No PyTorch model file found'})
        # Temporary 'path' request type for pytorch unittest
        elif 'path' in request.POST:
            config = torch.load(request.POST['path'])
        elif 'config' in request.POST:
            config = torch.load(request.POST['config'])
        elif 'url' in request.POST:
            try:
                url = urlparse(request.POST['url'])
                if url.netloc == 'github.com':
                    url = url._replace(netloc='raw.githubusercontent.com')
                    url = url._replace(path=url.path.replace('blob/', ''))
                config = torch.load(urllib2.urlopen(url.geturl()))
            except Exception as ex:
                return JsonResponse({'result': 'error', 'error': 'Invalid URL\n'+str(ex)})
        else:
            return JsonResponse({'result': 'error', 'error': 'No PyTorch model was found'})

    net = {}
    d = {}

    # Have to get input shape from user (Refer to pytorch docs)
    dummy_input = Variable(torch.randn(1, 8), requires_grad=True)
    operations = extract_nodes(config(dummy_input))

    while(operations):

        node, node_name = operations.popitem()
        """
        d[node] = {'type': [], 'input': [], 'output': [], 'params': {}}
        layer = d[node]['type']
        layer = layer_map(node)

        #Extract details from the classes and assign it to the dictionary

        # MmBackward = Dense without bias. AddmmBackward = Dense with bias

        if node_name == 'MmBackward' or node_name == 'AddmmBackward':
            W_Shape = /
            (str(list(node.next_functions[2][0].next_functions[0][0].variable.size())).replace('L',''))[1:-1]
            if node_name == 'AddmmBackward':
                Bias_Shape = /
                (str(list(node.next_functions[0][0].next_functions[0][0].variable.size())).replace('L',''))[1:-1]
        """

    for key in d.keys():
        net[key] = {
            'info': {
                'type': d[key]['type'][0],
                'phase': None
            },
            'connection': {
                'input': d[key]['input'],
                'output': d[key]['output']
            },
            'params': d[key]['params']
        }

    return JsonResponse({'result': 'success', 'net': net, 'net_name': ''})
