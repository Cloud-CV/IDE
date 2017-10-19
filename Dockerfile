FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils git \
                            libatlas-base-dev python-protobuf python-numpy \
                            python-scipy python-h5py unzip make libblas-dev \
                            liblapack-dev libatlas-base-dev gfortran \
                            python-pip python-dev libprotobuf-dev \
                            libleveldb-dev libsnappy-dev libopencv-dev \
                            libhdf5-serial-dev libgflags-dev nodejs npm \
                            libgoogle-glog-dev liblmdb-dev protobuf-compiler \
                            cmake libboost-all-dev wget python-setuptools g++ \
                            && rm -rf /var/lib/apt/lists/*

ADD requirements/dev.txt requirements/common.txt /tmp/

RUN pip install setuptools wheel \
    && pip install numpy scipy scikit-image \
    && pip install -r /tmp/dev.txt \
    && pip install --upgrade keras==2.0.7 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

RUN mkdir /caffe && cd /caffe \
    && wget https://github.com/BVLC/caffe/archive/25391bf9e0552740af8253c6d6fd484297889a49.zip \
    && unzip -o 25391bf9e0552740af8253c6d6fd484297889a49.zip \
    && rm 25391bf9e0552740af8253c6d6fd484297889a49.zip \
    && mv caffe-25391bf9e0552740af8253c6d6fd484297889a49 caffe \
    && cd caffe \
    && mkdir build && cd build/ \
    && cmake -DCPU_ONLY=4 -DWITH_PYTHON_LAYER=1 .. \
    && make -j 4

RUN ln -s /usr/bin/nodejs /usr/bin/node
