# Fabrik

[![Join the chat at https://gitter.im/Cloud-CV/IDE](https://badges.gitter.im/Cloud-CV/IDE.svg)](https://gitter.im/Cloud-CV/IDE?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/Cloud-CV/Fabrik.svg?branch=master)](https://travis-ci.org/Cloud-CV/Fabrik)
[![Coverage Status](https://coveralls.io/repos/github/Cloud-CV/Fabrik/badge.svg?branch=master)](https://coveralls.io/github/Cloud-CV/Fabrik?branch=coveralls)

This is a React+Django webapp with a simple drag and drop interface to build and configure deep neural networks with support for export of model configuration files to caffe and tensorflow. It also supports import from these frameworks to visualize different model architectures. Our motivation is to build an online IDE where researchers can share models and collaborate without having to deal with deep learning code.

### Interface
<img src="/example/fabrik_demo?raw=true.gif">

This app is presently under active development and we welcome contributions. Please check out our [issues thread](https://github.com/Cloud-CV/IDE/issues) to find things to work on, or ping us on [Gitter](https://gitter.im/Cloud-CV/IDE). 

### How to setup
1. First set up a virtualenv
    ```
    sudo apt-get install python-pip python-dev python-virtualenv 
    virtualenv --system-site-packages ~/Fabrik
    source ~/Fabrik/bin/activate
    ```
    
2. Clone the repository
    ```
    git clone --recursive https://github.com/Cloud-CV/Fabrik.git
    ```
    
3. If you have Caffe, Keras and Tensorflow already installed on your computer, skip this step
    * For Linux users
        ```
        cd Fabrik/requirements
        sh caffe_tensorflow_keras_install.sh
        ```
    * For Mac users
        * [Install Caffe](http://caffe.berkeleyvision.org/install_osx.html)
        * [Install Tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup#virtualenv_installation)
        * [Install Keras](https://keras.io/#installation)
4. Install dependencies
* For developers:
    ```
    pip install -r requirements/dev.txt
    ```
* Others:
    ```
    pip install -r requirements/common.txt
    ```
5. [Install postgres](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-16-04)
* Setup postgres database
    ```
      psql -c "CREATE DATABASE fabrik" -U postgres
      psql -c "CREATE USER admin WITH PASSWORD 'fabrik'" -U postgres
      psql -c "ALTER ROLE admin SET client_encoding TO 'utf8'" -U postgres
      psql -c "ALTER ROLE admin SET default_transaction_isolation TO 'read committed'" -U postgres
      psql -c "ALTER ROLE admin SET timezone TO 'UTC'" -U postgres
      psql -c "ALTER USER admin CREATEDB" -U postgres
    ```
* Migrate
    ```
    python manage.py makemigrations caffe_app
    python manage.py migrate
    ```
6. Install node modules
```
npm install
webpack --progress --watch --colors
```

### Usage
```
KERAS_BACKEND=theano python manage.py runserver
```

### Example
* Use `example/tensorflow/GoogleNet.pbtxt` for tensorflow import
* Use `example/caffe/GoogleNet.prototxt` for caffe import
* Use `example/keras/vgg16.json` for keras import

### License

This software is licensed under GNU GPLv3. Please see the included License file. All external libraries, if modified, will be mentioned below explicitly.
