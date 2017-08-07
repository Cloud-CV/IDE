from django.conf.urls import url
from views.import_prototxt import importPrototxt
from views.export_prototxt import exportToCaffe
from views.DB import saveToDB
from views.DB import loadFromDB

urlpatterns = [
    url(r'^export$', exportToCaffe, name='caffe-export'),
    url(r'^import$', importPrototxt, name='caffe-import'),
    url(r'^save$', saveToDB, name='saveDB'),
    url(r'^load*', loadFromDB, name='loadDB')
]
