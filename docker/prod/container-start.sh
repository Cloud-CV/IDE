#!/bin/sh
cd /code && \
webpack
python manage.py runserver 0.0.0.0:8000
