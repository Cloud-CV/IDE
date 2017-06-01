import json
import os
import unittest
from django.conf import settings
from django.core.urlresolvers import reverse
from django.test import Client


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        sample_file = open(os.path.join(settings.BASE_DIR, 'example', 'GoogleNet.prototxt'), 'r')
        response = self.client.post(reverse('caffe-import'), {'file': sample_file})
        response = json.loads(response.content)
        self.assertEqual(response['result'], 'success')
