import json
import os
import unittest
import torch
from torch.autograd import Variable
from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client
from pytorch_app.views.layers_import import extract_nodes


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_pt_import(self):
        sample_file = os.path.join(settings.BASE_DIR, 'example', 'pytorch', 'test.pt')
        response = self.client.post(reverse('pytorch-import'), {'path': sample_file})
        """
        # Ideally we would want to use something like below instead; refer to docs
        sample_file = open(os.path.join(settings.BASE_DIR, 'example', 'pytorch', 'test.pt'), 'rb')
        response = self.client.post(reverse('pytorch-import'), {'file': sample_file})
        """
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class GraphTraceTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_graph_trace(self):
        dummy_input = Variable(torch.randn(1, 8), requires_grad=True)
        model = torch.load(os.path.join(settings.BASE_DIR, 'example', 'pytorch', 'test.pt'))
        operations = extract_nodes(model(dummy_input))
        node, node_name = operations.popitem()
        self.assertEqual(node_name, 'AddmmBackward')
