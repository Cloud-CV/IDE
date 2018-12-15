from .common import * # flake8: noqa

# Database
# https://docs.djangoproject.com/en/1.9/ref/settings/#databases

DEBUG = False

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'fabrik', # Change this to 'postgres' if you're using docker
        'USER': 'admin', # Change this to 'postgres' if you're using docker
        'PASSWORD': 'fabrik', # Change this to 'postgres' if you're using docker
        'HOST': 'localhost', # Change this to 'db' if you're using docker
        'PORT': 5432,
    }
}

TEST = True
