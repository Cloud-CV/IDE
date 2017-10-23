python /code/manage.py makemigrations caffe_app
python /code/manage.py migrate

cd /code && npm install

/code/node_modules/.bin/webpack --progress

python /code/manage.py runserver 0.0.0.0:8080


