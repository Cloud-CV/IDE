import json
import os
import unittest
import yaml

from caffe import layers as L, params as P, to_proto
from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client
from ide.utils.jsonToPrototxt import jsonToPrototxt


# ********** Data Layers Test **********
class ImageDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data, label = L.ImageData(source='/dummy/source/', batch_size=32, ntop=2, rand_skip=0,
                                  shuffle=False, new_height=256, new_width=256, is_color=False,
                                  root_folder='/dummy/folder/',
                                  transform_param=dict(crop_size=227, mean_value=[104, 117, 123],
                                                       mirror=True, force_color=False,
                                                       force_gray=False))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'ImageData')


class DataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data, label = L.Data(source='/dummy/source/', backend=P.Data.LMDB, batch_size=32, ntop=2,
                             rand_skip=0, prefetch=10,
                             transform_param=dict(crop_size=227, mean_value=[104, 117, 123],
                                                  mirror=True, force_color=False,
                                                  force_gray=False))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'Data')


class HDF5DataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data, label = L.HDF5Data(source='/dummy/source/', batch_size=32, ntop=2, shuffle=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'HDF5Data')


class HDF5OutputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.HDF5Output(data, file_name='/dummy/filename')
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'HDF5Output')


class InputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'Input')


class WindowDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data, label = L.WindowData(source='/dummy/source/', batch_size=32, ntop=2,
                                   fg_threshold=0.5, bg_threshold=0.5, fg_fraction=0.25,
                                   context_pad=0, crop_mode='warp', cache_images=False,
                                   root_folder='/dummy/folder/',
                                   transform_param=dict(crop_size=227, mean_value=[104, 117, 123],
                                                        mirror=True, force_color=False,
                                                        force_gray=False))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'WindowData')


class MemoryDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data, label = L.MemoryData(batch_size=32, ntop=2, channels=3, height=224, width=224)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'MemoryData')


class DummyDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.DummyData(shape={'dim': [10, 3, 224, 224]},
                           data_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l0']['info']['type'], 'DummyData')


# ********** Vision Layers Test **********
class ConvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Convolution(data, kernel_size=3, pad=1, stride=1, num_output=128,
                            weight_filler={'type': 'xavier'}, bias_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Convolution')


class PoolingLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Pooling(data, kernel_size=2, pad=0, stride=2, pool=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Pooling')


class SPPLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.SPP(data, pyramid_height=2, pool=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'SPP')


class CropLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Crop(data, axis=2, offset=2)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Crop')


class DeconvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Deconvolution(data, convolution_param=dict(kernel_size=3, pad=1, stride=1,
                              num_output=128, weight_filler={'type': 'xavier'},
                              bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Deconvolution')


# ********** Recurrent Layers Test **********
class RecurrentLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Recurrent(data, recurrent_param=dict(num_output=128, debug_info=False,
                          expose_hidden=False, weight_filler={'type': 'xavier'},
                          bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Recurrent')


class RNNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.RNN(data, recurrent_param=dict(num_output=128, debug_info=False,
                    expose_hidden=False, weight_filler={'type': 'xavier'},
                    bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'RNN')


class LSTMLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.LSTM(data, recurrent_param=dict(num_output=128, debug_info=False,
                     expose_hidden=False, weight_filler={'type': 'xavier'},
                     bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'LSTM')


# ********** Common Layers Test **********
class InnerProductLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.InnerProduct(data, num_output=128, weight_filler={'type': 'xavier'},
                             bias_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'InnerProduct')


class DropoutLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Dropout(data, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Dropout')


class EmbedLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Embed(data, num_output=128, input_dim=2, bias_term=False,
                      weight_filler={'type': 'xavier'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Embed')


# ********** Normalisation Layers Test **********
class LRNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.LRN(data, local_size=5, alpha=1, beta=0.75, k=1, norm_region=1, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'LRN')


class MVNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.MVN(data, normalize_variance=True, eps=1e-9, across_channels=False, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'MVN')


class BatchNormLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.BatchNorm(data, use_global_stats=True, moving_average_fraction=0.999, eps=1e-5,
                          in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'BatchNorm')


# ********** Activation / Neuron Layers Test **********
class ReLULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.ReLU(data, negative_slope=0, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'ReLU')


class PReLULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.PReLU(data, channel_shared=False, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'PReLU')


class ELULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.ELU(data, alpha=1, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'ELU')


class SigmoidLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Sigmoid(data, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Sigmoid')


class TanHLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.TanH(data, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'TanH')


class AbsValLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.AbsVal(data, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'AbsVal')


class PowerLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Power(data, power=1.0, scale=1.0, shift=0.0, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Power')


class ExpLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Exp(data, base=-1.0, scale=1.0, shift=0.0, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Exp')


class LogLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Log(data, base=-1.0, scale=1.0, shift=0.0, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Log')


class BNLLLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.BNLL(data, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'BNLL')


class ThresholdLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Threshold(data, threshold=1.0, in_place=True)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Threshold')


class BiasLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Bias(data, axis=1, num_axes=1, filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Bias')


class ScaleLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Scale(data, bias_term=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Scale')


# ********** Utility Layers Test **********
class FlattenLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Flatten(data, axis=1, end_axis=-1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Flatten')


class ReshapeLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Reshape(data, shape={'dim': [2, -1]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Reshape')


class BatchReindexLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.BatchReindex(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'BatchReindex')


class SplitLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Split(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Split')


class ConcatLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Concat(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Concat')


class SliceLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Slice(data, axis=1, slice_dim=1, slice_point=[1, 2])
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Slice')


class EltwiseLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Eltwise(data, operation=2)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Eltwise')


class FilterLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Filter(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Filter')


# This layer is currently not supported as there is no bottom blob
'''class ParameterLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Parameter(data, shape={'dim': [10, 3, 224, 224]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'ImageData')'''


class ReductionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Reduction(data, operation=3, axis=0, coeff=1.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Reduction')


class SilenceLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Silence(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Silence')


class ArgMaxLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.ArgMax(data, out_max_val=False, top_k=1, axis=0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'ArgMax')


class SoftmaxLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Softmax(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Softmax')


# ********** Loss Layers Test **********
class MultinomialLogisticLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.MultinomialLogisticLoss(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'MultinomialLogisticLoss')


class InfogainLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.InfogainLoss(data, source='/dummy/source/', axis=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'InfogainLoss')


class SoftmaxWithLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.SoftmaxWithLoss(data, softmax_param=dict(axis=1))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'SoftmaxWithLoss')


class EuclideanLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.EuclideanLoss(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'EuclideanLoss')


class HingeLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.HingeLoss(data, norm=2)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'HingeLoss')


class SigmoidCrossEntropyLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.SigmoidCrossEntropyLoss(data)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'SigmoidCrossEntropyLoss')


class AccuracyLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.Accuracy(data, axis=1, top_k=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'Accuracy')


class ContrastiveLossLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        top = L.ContrastiveLoss(data, margin=1.0, legacy_version=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        net = yaml.safe_load(json.dumps(response['net']))
        prototxt, input_dim = jsonToPrototxt(net, response['net_name'])
        self.assertGreater(len(prototxt), 9)
        self.assertEqual(net['l1']['info']['type'], 'ContrastiveLoss')
