import json
import os
import unittest

from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import SimpleRNN, LSTM
from keras.layers import Embedding
from keras.layers import add, concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import BatchNormalization

from keras.models import Model, Sequential


class ImportJsonTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        sample_file = open(os.path.join(settings.BASE_DIR, 'example', 'vgg16.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')


# ********** Data Layers **********
class InputLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Input((224, 224, 3))
        model = Model(model, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[0]]['params']), 1)


# ********** Vision Layers **********
class ConvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = Conv2D(64, (3, 3), padding='same')(img_input)
        model = Activation('relu')(model)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[2]]['params']), 9)
        self.assertEqual(response['net'][layerId[0]]['info']['type'], 'ReLU')


class DeconvolutionLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = Conv2DTranspose(64, (3, 3), padding='same')(img_input)
        model = LeakyReLU()(model)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[2]]['params']), 9)
        self.assertEqual(response['net'][layerId[0]]['info']['type'], 'ReLU')


class PoolingLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = MaxPooling2D((2, 2), strides=(2, 2))(img_input)
        model = AveragePooling2D((2, 2), strides=(2, 2))(model)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[1]]['params']), 7)
        self.assertGreaterEqual(len(response['net'][layerId[2]]['params']), 7)


# ********** Common Layers **********
class DenseLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = Flatten()(img_input)
        model = Dense(100)(model)
        model = PReLU()(model)
        model = Dropout(0.5)(model)
        model = Reshape((1, 1, 100))(model)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[0]]['params']), 3)
        self.assertEqual(response['net'][layerId[1]]['info']['type'], 'Flatten')
        self.assertEqual(response['net'][layerId[2]]['info']['type'], 'PReLU')
        self.assertEqual(response['net'][layerId[3]]['info']['type'], 'Reshape')
        self.assertEqual(response['net'][layerId[4]]['info']['type'], 'Dropout')


class EmbeddingLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((100,))
        model = Embedding(100, 1000)(img_input)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[1]]['params']), 3)


# ********** Recurrent Layers **********
class RecurrentLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Embedding(100, output_dim=256))
        model.add(LSTM(32, return_sequences=True))
        model.add(SimpleRNN(64))
        model.build()
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[0]]['params']), 3)
        self.assertGreaterEqual(len(response['net'][layerId[0]]['params']), 3)


# ********** Normalisation Layers **********
class BatchNormLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = BatchNormalization(center=True, scale=True)(img_input)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[0]]['info']['type'], 'Scale')
        self.assertEqual(response['net'][layerId[1]]['info']['type'], 'BatchNorm')


# ********** Utility Layers **********
class PaddingLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = ZeroPadding2D((3, 3))(img_input)
        model = Conv2D(64, (7, 7), strides=(2, 2))(model)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[0]]['params']['pad_h'], 3)


class EltwiseLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 64))
        model = Conv2D(64, (3, 3), padding='same')(img_input)
        model = add([img_input, model])
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[1]]['params']['operation'], 1)


class ConcatLayerTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = Conv2D(64, (3, 3), padding='same')(img_input)
        model = concatenate([img_input, model])
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = response['net'].keys()
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[2]]['info']['type'], 'Concat')
