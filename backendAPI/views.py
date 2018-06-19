# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.generic import TemplateView
from django.http import JsonResponse


# Create your views here.
class login(TemplateView):
    template_name = 'login.html'


def check_login(request):
    user = request.user
    user_id = user.id
    username = ''
    is_authenticated = request.user.is_authenticated()
    if (is_authenticated):
        username = user.username
    return JsonResponse({
        'result': is_authenticated,
        'user_id': user_id,
        'username': username
    })
