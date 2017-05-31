from django.test import Client
from django.conf import settings
import unittest
import json
import os

class UploadTest(unittest.TestCase):
		def setUp(self):
				self.client = Client()

		def test_details(self):
				sample_file = open(os.path.join(settings.BASE_DIR,'example','GoogleNet.prototxt'), 'r')
				response = self.client.post('/caffe/import', {'file': sample_file})
				response = json.loads(response.content)
				print 'Test {:}!'.format(response['result'])
