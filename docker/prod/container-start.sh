#!/bin/sh
cd /code && \
webpack
KERAS_BACKEND=theano uwsgi --ini /code/docker/prod/uwsgi.ini
