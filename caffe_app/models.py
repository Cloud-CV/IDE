from __future__ import unicode_literals

from django.db import models
from django.contrib.postgres.fields import JSONField


class ModelExport(models.Model):
    name = models.CharField(max_length=100)
    id = models.CharField(primary_key=True)
    network = JSONField()
