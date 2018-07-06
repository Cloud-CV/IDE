python /code/manage.py makemigrations caffe_app --settings=ide.docker_settings
python /code/manage.py migrate --settings=ide.docker_settings
KERAS_BACKEND=theano coverage run --source=/code/caffe_app,/code/keras_app,/code/ide/utils /code/manage.py test --settings=ide.docker_settings
python /code/manage.py runserver 0.0.0.0:8080 --settings=ide.docker_settings
