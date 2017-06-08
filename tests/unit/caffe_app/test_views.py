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
        with open(os.path.join(settings.BASE_DIR, 'media', 'imageData_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'imageData_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'imageData_test.prototxt'))
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
        with open(os.path.join(settings.BASE_DIR, 'media', 'data_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'data_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'data_test.prototxt'))
        self.assertEqual(response['result'], 'success')


class HDF5DataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data, label = L.HDF5Data(source='/dummy/source/', batch_size=32, ntop=2, shuffle=False)
        with open(os.path.join(settings.BASE_DIR, 'media', 'hdf5_data_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'hdf5_data_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'hdf5_data_test.prototxt'))
        self.assertEqual(response['result'], 'success')


class HDF5OutputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        top = L.HDF5Output(file_name='/dummy/filename')
        with open(os.path.join(settings.BASE_DIR, 'media', 'hdf5_output_test.prototxt'), 'w') as f:
            f.write(str(to_proto(top)))
        sample_file = open(os.path.join(
            settings.BASE_DIR, 'media', 'hdf5_output_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'hdf5_output_test.prototxt'))
        self.assertEqual(response['result'], 'success')


class InputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data = L.Input(shape={'dim': [10, 3, 224, 224]})
        with open(os.path.join(settings.BASE_DIR, 'media', 'input_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'input_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'input_test.prototxt'))
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
        with open(os.path.join(settings.BASE_DIR, 'media', 'window_data_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(
            settings.BASE_DIR, 'media', 'window_data_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'window_data_test.prototxt'))
        self.assertEqual(response['result'], 'success')


class MemoryDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data, label = L.MemoryData(batch_size=32, ntop=2, channels=3, height=224, width=224)
        with open(os.path.join(settings.BASE_DIR, 'media', 'memory_data_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data, label)))
        sample_file = open(os.path.join(
            settings.BASE_DIR, 'media', 'memory_data_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'memory_data_test.prototxt'))
        self.assertEqual(response['result'], 'success')


class DummyDataLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        data = L.DummyData(shape={'dim': [10, 3, 224, 224]},
                           data_filler={'type': 'constant'})
        with open(os.path.join(settings.BASE_DIR, 'media', 'dummy_data_test.prototxt'), 'w') as f:
            f.write(str(to_proto(data)))
        sample_file = open(os.path.join(
            settings.BASE_DIR, 'media', 'dummy_data_test.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        os.remove(os.path.join(settings.BASE_DIR, 'media', 'dummy_data_test.prototxt'))
        self.assertEqual(response['result'], 'success')
