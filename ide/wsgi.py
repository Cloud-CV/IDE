"""
WSGI config for ide project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.9/howto/deployment/wsgi/
"""
# import os module
import os
# import get wsgi application 
# on django core
from django.core.wsgi import get_wsgi_application
# set os environ to default with is ide settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ide.settings")
# create the application with is our imported
# django core, get wsgi application
application = get_wsgi_application()
