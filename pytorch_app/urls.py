from django.conf.urls import url
from views.import_pt import import_model
from views.export_pt import export_model

urlpatterns = [
    url(r'^import$', import_model, name='pytorch-import'),
    url(r'^export$', export_model, name='pytorch-export')
]
