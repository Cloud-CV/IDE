from .common import * # noqa: ignore=F405
import os

# Database
# https://docs.djangoproject.com/en/1.9/ref/settings/#databases

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': os.environ.get("POSTGRES_NAME", 'postgres'),
        'USER': os.environ.get("POSTGRES_USER", 'admin'),
        'PASSWORD': os.environ.get("POSTGRES_PASSWORD", 'fabrik'),
        'HOST': os.environ.get("POSTGRES_HOST", 'localhost'),
        'PORT': os.environ.get("POSTGRES_PORT", 5432),
    }
}
