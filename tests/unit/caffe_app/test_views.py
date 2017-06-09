import json
import os
import unittest
from caffe import layers as L, params as P, to_proto
from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client


class ImportPrototxtTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        sample_file = open(os.path.join(settings.BASE_DIR, 'example', 'GoogleNet.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


# ********** Data Layers Test **********
class ImageDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
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
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class DataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
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
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class HDF5DataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data, label = L.HDF5Data(source='/dummy/source/', batch_size=32, ntop=2, shuffle=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class HDF5OutputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.HDF5Output(file_name='/dummy/filename')
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class InputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class WindowDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
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
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class MemoryDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data, label = L.MemoryData(batch_size=32, ntop=2, channels=3, height=224, width=224)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class DummyDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data = L.DummyData(shape={'dim': [10, 3, 224, 224]},
                           data_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Vision Layers Test **********
class ConvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Convolution(kernel_size=3, pad=1, stride=1, num_output=128,
                            weight_filler={'type': 'xavier'}, bias_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class PoolingLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Pooling(kernel_size=2, pad=0, stride=2, pool=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SPPLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.SPP(pyramid_height=2, pool=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class CropLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Crop(axis=2, offset=2)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class DeconvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Deconvolution(convolution_param=dict(kernel_size=3, pad=1, stride=1, num_output=128,
                              weight_filler={'type': 'xavier'}, bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Recurrent Layers Test **********
class RecurrentLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Recurrent(recurrent_param=dict(num_output=128, debug_info=False,
                          expose_hidden=False, weight_filler={'type': 'xavier'},
                          bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class RNNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.RNN(recurrent_param=dict(num_output=128, debug_info=False,
                    expose_hidden=False, weight_filler={'type': 'xavier'},
                    bias_filler={'type': 'constant'}))
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Common Layers Test **********
class InnerProductLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.InnerProduct(num_output=128, weight_filler={'type': 'xavier'},
                             bias_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class DropoutLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Dropout()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class EmbedLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Embed(num_output=128, input_dim=2, bias_term=False,
                      weight_filler={'type': 'xavier'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Normalisation Layers Test **********
class LRNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.LRN(local_size=5, alpha=1, beta=0.75, k=1, norm_region=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class MVNLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.MVN(normalize_variance=True, eps=1e-9, across_channels=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class BatchNormLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.BatchNorm(use_global_stats=True, moving_average_fraction=0.999, eps=1e-5)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Activation / Neuron Layers Test **********
class ReLULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.ReLU(negative_slope=0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class PReLULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.PReLU(channel_shared=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ELULayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.ELU(alpha=1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SigmoidLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Sigmoid()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class TanHLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.TanH()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class AbsValLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.AbsVal()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class PowerLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Power(power=1.0, scale=1.0, shift=0.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ExpLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Exp(base=-1.0, scale=1.0, shift=0.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class LogLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Log(base=-1.0, scale=1.0, shift=0.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class BNLLLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.BNLL()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ThresholdLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Threshold(threshold=1.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class BiasLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Bias(axis=1, num_axes=1, filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ScaleLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Scale(bias_term=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


# ********** Utility Layers Test **********
class FlattenLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Flatten(axis=1, end_axis=-1)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ReshapeLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Reshape(shape={'dim': [2, -1]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class BatchReindexLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.BatchReindex()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SplitLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Split()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SplitLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Split()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ConcatLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Concat()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SliceLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Slice(axis=1, slice_dim=1, slice_point=[1, 2])
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class EltwiseLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Eltwise(operation=2)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class FilterLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Filter()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ParameterLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Parameter(shape={'dim': [10, 3, 224, 224]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ReductionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Reduction(operation=3, axis=0, coeff=1.0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SilenceLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Silence()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class ArgMaxLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.ArgMax(out_max_val=False, top_k=1, axis=0)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')


class SoftmaxLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.Softmax()
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'test.prototxt'))
        self.assertEqual(response['result'], 'success')
