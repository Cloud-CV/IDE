import json
import os
import unittest
from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client
from django.contrib.auth.models import User
from caffe_app.models import Network, NetworkVersion


class SaveToDBTest(unittest.TestCase):

    def setUp(self):
        self.client = Client()

    def test_save_json(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'ide',
                                  'caffe_export_test.json'), 'r')
        net = json.load(tests)['net']
        response = self.client.post(
            reverse('saveDB'),
            {'net': net, 'net_name': 'netname'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')

    def test_load(self):
        u_1 = User(id=1, username='user_1')
        u_1.save()
        u_2 = User(id=2, username='user_2')
        u_2.save()
        model = Network(name='net')
        model.save()
        model_version = NetworkVersion(network=model, network_def={})
        model_version.save()

        response = self.client.post(
            reverse('saveDB'),
            {'net': '{"net": "testnet"}', 'net_name': 'name'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertTrue('id' in response)
        proto_id = response['id']
        response = self.client.post(reverse('loadDB'), {'proto_id': proto_id})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net_name'], 'name')

    def test_load_nofile(self):
        response = self.client.post(reverse('loadDB'),
                                    {'proto_id': 'inexistent'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'error')
        self.assertEqual(response['error'], 'No network file found')


class SaveModelToDBTest(unittest.TestCase):

    def setUp(self):
        self.client = Client()

    def test_save_json1(self):
        tests = open(os.path.join(settings.BASE_DIR, 'tests', 'unit', 'ide',
                                  'caffe_export_test.json'), 'r')
        net = json.load(tests)['net']
        response = self.client.post(
            reverse('saveModel'),
            {'net': net, 'net_name': 'netname'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')

    def test_load1(self):
        u_3 = User(id=3, username='user_3')
        u_3.save()
        u_4 = User(id=4, username='user_4')
        u_4.save()
        model = Network(name='net')
        model.save()
        model_version = NetworkVersion(network=model, network_def={})
        model_version.save()

        response = self.client.post(
            reverse('saveModel'),
            {'net': '{"net": "testnet"}', 'net_name': 'name'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertTrue('id' in response)
        proto_id = response['id']
        response = self.client.post(reverse('loadDB'), {'proto_id': proto_id})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['net_name'], 'name')

    def test_load_nofile1(self):
        response = self.client.post(reverse('loadDB'),
                                    {'proto_id': 'inexistent'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'error')
        self.assertEqual(response['error'], 'No network file found')


class LoadModelFromDB(unittest.TestCase):

    def setUp(self):
        self.client = Client()

    def test_load_model(self):
        u_5 = User(id=5, username='user_5')
        u_5.save()
        model = Network(id=9, name='test_net', author_id='5')
        model.save()
        response = self.client.post(
            reverse('getModelData'), {'userID': '5'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertEqual(response['data']['Model1_Name'], 'test_net')
        self.assertEqual(response['data']['Model1_ID'], 9)


class DeleteModelFromDB(unittest.TestCase):

    def setUp(self):
        self.client = Client()

    def test_delete_model(self):
        u_6 = User(id=6, username='user_6')
        u_6.save()
        model = Network(id=10, name='test_net2', author_id='6')
        model.save()
        response = self.client.post(
            reverse('deleteModel'), {'userID': '6', 'modelid': '10'})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
        self.assertEqual(Network.objects.filter(id=10).exists(), False)
