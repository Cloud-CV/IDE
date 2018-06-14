from __future__ import unicode_literals

from django.db import models
from django.contrib.postgres.fields import JSONField
from django.contrib.auth.models import User


class Network(models.Model):
    name = models.CharField(max_length=100)
    network = JSONField()
    author = models.ForeignKey(User, blank=True, null=True)
    publicSharing = models.BooleanField(default=False)
    createdOn = models.DateField(auto_now_add=True)
    updatedOn = models.DateField(auto_now_add=True)

    def __unicode__(self):
        return self.name


class SharedWith(models.Model):
    ACCESS_PRIVILEGE = (
        ('E', 'Can Edit'),
        ('V', 'Can View'),
        ('C', 'Can Comment')
    )
    network = models.ForeignKey(Network)
    user = models.ForeignKey(User)
    access_privilege = models.CharField(max_length=1, choices=ACCESS_PRIVILEGE)
    createdOn = models.DateField(auto_now_add=True)
    updatedOn = models.DateField(auto_now_add=True)

    def __unicode__(self):
        return self.user.username
