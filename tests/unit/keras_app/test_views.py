import json
import os
import unittest
import yaml

from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector
from keras.layers import ActivityRegularization, Masking
from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose
from keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from keras.layers import LocallyConnected1D, LocallyConnected2D
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Embedding
from keras.layers import add, multiply, maximum, concatenate, average, dot
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise, GaussianDropout, AlphaDropout
from keras.layers import Input
from keras import regularizers
from keras.models import Model, Sequential
from keras_app.views.layers_export import data, convolution, deconvolution, pooling, dense, dropout, embed,\
    depthwiseConv, recurrent, batchNorm, activation, flatten, reshape, eltwise, concat, upsample,\
    locallyConnected, permute, repeatVector, regularization, masking, gaussianNoise, gaussianDropout,\
    alphaDropout


class ImportJsonTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # Test 1
        sample_file = open(os.path.join(settings.BASE_DIR, 'example/keras', 'vgg16.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        # Test 2
        sample_file = open(os.path.join(settings.BASE_DIR, 'example/caffe', 'GoogleNet.prototxt'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'error')
        self.assertEqual(response['error'], 'Invalid JSON')


class ExportJsonTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_export(self):
        # Test 1
        img_input = Input((224, 224, 3))
        model = Conv2D(64, (3, 3), padding='same', dilation_rate=1, use_bias=True,
                       kernel_regularizer=regularizers.l1(), bias_regularizer='l1',
                       activity_regularizer='l1', kernel_constraint='max_norm',
                       bias_constraint='max_norm')(img_input)
        model = Model(img_input, model)
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        response = self.client.post(reverse('keras-export'), {'net': json.dumps(response['net']),
                                                              'net_name': ''})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        # Test 2
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'ide',
                                  'caffe_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['HDF5Data']}
        response = self.client.post(reverse('keras-export'), {'net': json.dumps(net),
                                                              'net_name': ''})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'error')


# ********** Import json tests **********
class HelperFunctions():
    def setUp(self):
        self.client = Client()

    def keras_type_test(self, model, id, type):
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = sorted(response['net'].keys())
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[id]]['info']['type'], type)

    def keras_param_test(self, model, id, params):
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = sorted(response['net'].keys())
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[id]]['params']), params)


# ********** Data Layers **********
class InputImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Input((224, 224, 3))
        model = Model(model, model)
        self.keras_param_test(model, 0, 1)


# ********** Core Layers **********
class DenseImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                        bias_constraint='max_norm', activation='relu', input_shape=(16,)))
        model.build()
        self.keras_param_test(model, 1, 3)


class ActivationImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # softmax
        model = Sequential()
        model.add(Activation('softmax', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'Softmax')
        # relu
        model = Sequential()
        model.add(Activation('relu', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'ReLU')
        # tanh
        model = Sequential()
        model.add(Activation('tanh', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'TanH')
        # sigmoid
        model = Sequential()
        model.add(Activation('sigmoid', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'Sigmoid')
        # selu
        model = Sequential()
        model.add(Activation('selu', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'SELU')
        # softplus
        model = Sequential()
        model.add(Activation('softplus', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'Softplus')
        # softsign
        model = Sequential()
        model.add(Activation('softsign', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'Softsign')
        # hard_sigmoid
        model = Sequential()
        model.add(Activation('hard_sigmoid', input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'HardSigmoid')
        # LeakyReLU
        model = Sequential()
        model.add(LeakyReLU(alpha=1, input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'ReLU')
        # PReLU
        model = Sequential()
        model.add(PReLU(input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'PReLU')
        # ELU
        model = Sequential()
        model.add(ELU(alpha=1, input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'ELU')
        # ThresholdedReLU
        model = Sequential()
        model.add(ThresholdedReLU(theta=1, input_shape=(15,)))
        model.build()
        self.keras_type_test(model, 0, 'ThresholdedReLU')


class DropoutImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Dropout(0.5, input_shape=(10, 64)))
        model.build()
        self.keras_type_test(model, 0, 'Dropout')


class FlattenImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Flatten(input_shape=(10, 64)))
        model.build()
        self.keras_type_test(model, 0, 'Flatten')


class ReshapeImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Reshape((5, 2), input_shape=(10,)))
        model.build()
        self.keras_type_test(model, 0, 'Reshape')


class PermuteImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        model.build()
        self.keras_type_test(model, 0, 'Permute')


class RepeatVectorImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(RepeatVector(3, input_shape=(10,)))
        model.build()
        self.keras_type_test(model, 0, 'RepeatVector')


class ActivityRegularizationImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(ActivityRegularization(l1=2, input_shape=(10,)))
        model.build()
        self.keras_type_test(model, 0, 'Regularization')


class MaskingImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(5, 100)))
        model.build()
        self.keras_type_test(model, 0, 'Masking')


# ********** Convolutional Layers **********
class ConvolutionImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # Conv 1D
        model = Sequential()
        model.add(Conv1D(32, 3, kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                         bias_constraint='max_norm', activation='relu', input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 1, 9)
        # Conv 2D
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                         bias_constraint='max_norm', activation='relu', input_shape=(1, 16, 16)))
        model.build()
        self.keras_param_test(model, 1, 13)
        # Conv 3D
        model = Sequential()
        model.add(Conv3D(32, (3, 3, 3), kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                         bias_constraint='max_norm', activation='relu', input_shape=(1, 16, 16, 16)))
        model.build()
        self.keras_param_test(model, 1, 17)


# This is currently unavailable with Theano backend
'''
class DepthwiseConvolutionImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(SeparableConv2D(32, 3, depthwise_regularizer=regularizers.l2(0.01),
                                  pointwise_regularizer=regularizers.l2(0.01),
                                  bias_regularizer=regularizers.l2(0.01),
                                  activity_regularizer=regularizers.l2(0.01), depthwise_constraint='max_norm',
                                  bias_constraint='max_norm', pointwise_constraint='max_norm',
                                  activation='relu', input_shape=(1, 16, 16)))
        self.keras_param_test(model, 1, 12)'''


class DeconvolutionImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Conv2DTranspose(32, (3, 3), kernel_regularizer=regularizers.l2(0.01),
                                  bias_regularizer=regularizers.l2(0.01),
                                  activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                                  bias_constraint='max_norm', activation='relu', input_shape=(1, 16, 16)))
        model.build()
        self.keras_param_test(model, 1, 13)


class UpsampleImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # Upsample 1D
        model = Sequential()
        model.add(UpSampling1D(size=2, input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 2)
        # Upsample 2D
        model = Sequential()
        model.add(UpSampling2D(size=(2, 2), input_shape=(1, 16, 16)))
        model.build()
        self.keras_param_test(model, 0, 3)
        # Upsample 3D
        model = Sequential()
        model.add(UpSampling3D(size=(2, 2, 2), input_shape=(1, 16, 16, 16)))
        model.build()
        self.keras_param_test(model, 0, 4)


# ********** Pooling Layers **********
class PoolingImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # Global Pooling 1D
        model = Sequential()
        model.add(GlobalMaxPooling1D(input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 5)
        # Global Pooling 2D
        model = Sequential()
        model.add(GlobalMaxPooling2D(input_shape=(1, 16, 16)))
        model.build()
        self.keras_param_test(model, 0, 8)
        # Pooling 1D
        model = Sequential()
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 5)
        # Pooling 2D
        model = Sequential()
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', input_shape=(1, 16, 16)))
        model.build()
        self.keras_param_test(model, 0, 8)
        # Pooling 3D
        model = Sequential()
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same',
                               input_shape=(1, 16, 16, 16)))
        model.build()
        self.keras_param_test(model, 0, 11)


# ********** Locally-connected Layers **********
class LocallyConnectedImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        # Conv 1D
        model = Sequential()
        model.add(LocallyConnected1D(32, 3, kernel_regularizer=regularizers.l2(0.01),
                                     bias_regularizer=regularizers.l2(0.01),
                                     activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                                     bias_constraint='max_norm', activation='relu', input_shape=(10, 16)))
        model.build()
        self.keras_param_test(model, 1, 12)
        # Conv 2D
        model = Sequential()
        model.add(LocallyConnected2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01),
                                     bias_regularizer=regularizers.l2(0.01),
                                     activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                                     bias_constraint='max_norm', activation='relu', input_shape=(10, 16, 16)))
        model.build()
        self.keras_param_test(model, 1, 14)


# ********** Recurrent Layers **********
class RecurrentImportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
        model.add(SimpleRNN(32, return_sequences=True))
        model.add(GRU(10, kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l2(0.01), kernel_constraint='max_norm',
                      bias_constraint='max_norm', recurrent_constraint='max_norm'))
        model.build()
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = sorted(response['net'].keys())
        self.assertEqual(response['result'], 'success')
        self.assertGreaterEqual(len(response['net'][layerId[1]]['params']), 7)
        self.assertGreaterEqual(len(response['net'][layerId[3]]['params']), 7)
        self.assertGreaterEqual(len(response['net'][layerId[6]]['params']), 7)


# ********** Embedding Layers **********
class EmbeddingImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(Embedding(1000, 64, input_length=10, embeddings_regularizer=regularizers.l2(0.01),
                            embeddings_constraint='max_norm'))
        model.build()
        self.keras_param_test(model, 0, 7)


# ********** Merge Layers **********
class ConcatImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 3))
        model = Conv2D(64, (3, 3), padding='same')(img_input)
        model = concatenate([img_input, model])
        model = Model(img_input, model)
        self.keras_type_test(model, 0, 'Concat')


class EltwiseImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        img_input = Input((224, 224, 64))
        model = Conv2D(64, (3, 3), padding='same')(img_input)
        model = add([img_input, model])
        model = Model(img_input, model)
        self.keras_type_test(model, 0, 'Eltwise')


# ********** Normalisation Layers **********
class BatchNormImportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(BatchNormalization(center=True, scale=True, beta_regularizer=regularizers.l2(0.01),
                                     gamma_regularizer=regularizers.l2(0.01),
                                     beta_constraint='max_norm', gamma_constraint='max_norm',
                                     input_shape=(10, 16)))
        model.build()
        json_string = Model.to_json(model)
        with open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'w') as out:
            json.dump(json.loads(json_string), out, indent=4)
        sample_file = open(os.path.join(settings.BASE_DIR, 'media', 'test.json'), 'r')
        response = self.client.post(reverse('keras-import'), {'file': sample_file})
        response = json.loads(response.content)
        layerId = sorted(response['net'].keys())
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[0]]['info']['type'], 'Scale')
        self.assertEqual(response['net'][layerId[1]]['info']['type'], 'BatchNorm')


# ********** Noise Layers **********
class GaussianNoiseImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(GaussianNoise(stddev=0.1, input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 1)


class GaussianDropoutImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(GaussianDropout(rate=0.5, input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 1)


class AlphaDropoutImportTest(unittest.TestCase, HelperFunctions):
    def setUp(self):
        self.client = Client()

    def test_keras_import(self):
        model = Sequential()
        model.add(AlphaDropout(rate=0.5, seed=5, input_shape=(1, 16)))
        model.build()
        self.keras_param_test(model, 0, 1)


# ********** Utility Layers **********
class PaddingImportTest(unittest.TestCase):
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
        layerId = sorted(response['net'].keys())
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net'][layerId[0]]['params']['pad_h'], 3)


# ********** Export json tests **********

# ********** Data Layers Test **********
class InputExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input']}
        net = data(net['l0'], '', 'l0')
        model = Model(net['l0'], net['l0'])
        self.assertEqual(model.layers[0].__class__.__name__, 'InputLayer')


# ********** Vision Layers Test **********
class ConvolutionExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Convolution']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = convolution(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Conv2D')


class PoolingExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Pooling']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = pooling(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'AveragePooling2D')


class DeconvolutionExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Deconvolution']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = deconvolution(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Conv2DTranspose')


# ********** Recurrent Layers Test **********
class RNNExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input2'], 'l1': net['RNN']}
        net['l0']['connection']['output'].append('l1')
        # # net = get_shapes(net)
        inp = data(net['l0'], '', 'l0')['l0']
        net = recurrent(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'SimpleRNN')


class LSTMExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input2'], 'l1': net['LSTM']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = recurrent(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'LSTM')


# ********** Common Layers Test **********
class DenseExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input3'], 'l1': net['InnerProduct']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = dense(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Dense')


class DropoutExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input3'], 'l1': net['Dropout']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = dropout(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Dropout')


class EmbedExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input3'], 'l1': net['Embed']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = embed(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Embedding')


# ********** Normalisation Layers Test **********
class BatchNormExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['BatchNorm'], 'l2': net['Scale']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = batchNorm(net['l1'], [inp], 'l1', 'l2', net['l2'])
        model = Model(inp, net['l2'])
        self.assertEqual(model.layers[1].__class__.__name__, 'BatchNormalization')


# ********** Activation / Neuron Layers Test **********
class ReLUExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['ReLU']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Activation')


class PReLUExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['PReLU']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'PReLU')


class ELUExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['ELU']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'ELU')


class SigmoidExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Sigmoid']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Activation')


class TanHExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['TanH']}
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Activation')


# ********** Utility Layers Test **********
class FlattenExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Flatten']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = flatten(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Flatten')


class ReshapeExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Reshape']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = reshape(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Reshape')


class ConcatExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Concat']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = concat(net['l1'], [inp, inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Concatenate')


class EltwiseExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Eltwise']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = eltwise(net['l1'], [inp, inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Maximum')


class SoftmaxExportTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_json_to_prototxt(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'keras_app',
                                  'keras_export_test.json'), 'r')
        response = json.load(tests)
        tests.close()
        net = yaml.safe_load(json.dumps(response['net']))
        net = {'l0': net['Input'], 'l1': net['Softmax']}
        net['l0']['connection']['output'].append('l1')
        inp = data(net['l0'], '', 'l0')['l0']
        net = activation(net['l1'], [inp], 'l1')
        model = Model(inp, net['l1'])
        self.assertEqual(model.layers[1].__class__.__name__, 'Activation')
