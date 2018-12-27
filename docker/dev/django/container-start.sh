#!/bin/sh
cd /code && \
python manage.py makemigrations caffe_app && \
python manage.py migrate --noinput && \
python manage.py runserver 0.0.0.0:8000
