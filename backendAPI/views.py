# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.generic import TemplateView
from django.http import JsonResponse
from caffe_app.models import ModelExport
# from django.shortcuts import render
# from django.core import serializers


# Create your views here.
class login(TemplateView):
    template_name = 'home.html'


def checkLogin(request):
    return JsonResponse({'result': request.user.is_authenticated()})


def getModels(request):
    if (request.user.is_authenticated()):
        models = ModelExport.objects.filter(user=request.user).values('name', 'id')
        #
        return JsonResponse(list(models), safe=False)
