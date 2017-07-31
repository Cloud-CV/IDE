#!/bin/sh
cd /code && \
webpack
KERAS_BACKEND=theano uwsgi --ini ide_uwsgi.ini
