import json
import os
import unittest

from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        sample_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'GoogleNet.pbtxt'),
                           'r')
        response = self.client.post(reverse('tf-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class ConvLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'Conv3DCheck.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class DeconvLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'denoiseAutoEncoder.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class PoolLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'Pool3DCheck.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class RepeatLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'Conv2DRepeat.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class StackLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'FCStack.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class DepthwiseConvLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'DepthwiseConv.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class UpsampleLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'UNet.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class BatchNormLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'BatchNorm.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class GRUCellTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'GRUCell.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class LSTMCellTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'LSTMCell.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


class RNNCellTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_tf_import(self):
        model_file = open(os.path.join(settings.BASE_DIR, 'example/tensorflow', 'RNNCell.pbtxt'),
                          'r')
        response = self.client.post(reverse('tf-import'), {'file': model_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
