from django.conf.urls import url
from views.import_pytorch import import_model


urlpatterns = [
    url(r'^import$', import_model, name='pytorch-import'),
]
