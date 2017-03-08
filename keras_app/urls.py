from django.conf.urls import url
from views.import_keras import import_keras_json
from views.export_keras import export_keras_json

urlpatterns = [
    url(r'^export$', export_keras_json),
    url(r'^import$', import_keras_json),
]
