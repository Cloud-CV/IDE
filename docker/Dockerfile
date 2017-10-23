FROM bvlc/caffe:intel

RUN wget -qO- https://deb.nodesource.com/setup_4.x | bash - \
    && apt-get install -y --no-install-recommends apt-utils \
                            libatlas-base-dev python-protobuf \
                            python-h5py unzip make libblas-dev \
                            liblapack-dev libatlas-base-dev gfortran \
                            nodejs python-pip python-setuptools libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip2.7 install setuptools wheel \
     && pip2.7 install numpy scipy scikit-image \
     && pip2.7 install Django==1.9.3 pyaml==15.8.2 django-allauth==0.30.0 \
                       psycopg2==2.7.3 pep8==1.7.0 flake8==3.0.4 \
                       python-coveralls==2.9.1 \
     && pip2.7 install --upgrade keras==2.0.7 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
