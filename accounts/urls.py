from django.conf.urls import url
from views import signup, login

urlpatterns = [
    url(r'^signup', signup),
    url(r'^login', login)
]
