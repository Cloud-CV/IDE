from django.conf.urls import url
from views import login, checkLogin, getModels

urlpatterns = [
    url(r'^login$', login.as_view()),
    url(r'^checkLogin$', checkLogin),
    url(r'^getModels$', getModels)
]
