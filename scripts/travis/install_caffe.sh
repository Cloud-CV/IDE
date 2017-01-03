#!/bin/bash
# This script must be run with sudo.

MAKE="make --jobs=$NUM_THREADS"

# This ppa is for gflags and glog
sudo add-apt-repository -y ppa:tuleu/precise-backports
sudo apt-get -y update
sudo apt-get install -y \
    wget git curl \
    python-dev python-numpy python3-dev\
    libleveldb-dev libsnappy-dev libopencv-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-dev libatlas-base-dev \
    libhdf5-serial-dev libgflags-dev libgoogle-glog-dev \
    bc

echo "Installed dependencies"

# Add a special apt-repository to install CMake 2.8.9 for CMake Caffe build,
# if needed.  By default, Aptitude in Ubuntu 12.04 installs CMake 2.8.7, but
# Caffe requires a minimum CMake version of 2.8.8.
A=$PWD
cd $TOOLS_DIR
if $WITH_CMAKE; then
  # cmake 3 will make sure that the python interpreter and libraries match
  wget --no-check-certificate http://www.cmake.org/files/v3.2/cmake-3.2.3-Linux-x86_64.sh -O cmake3.sh
  chmod +x cmake3.sh
  ./cmake3.sh --prefix=$TOOLS_DIR --skip-license --exclude-subdir
fi

conda install --yes numpy scipy matplotlib scikit-image pip
# Let conda install boost (so that boost_python matches)
conda install --yes -c https://conda.binstar.org/menpo boost=1.56.0

echo "Installed conda packages and cmake"

# Install LMDB
LMDB_URL=https://github.com/LMDB/lmdb/archive/LMDB_0.9.18.tar.gz
LMDB_FILE=$TOOLS_DIR/lmdb.tar.gz
pushd .
wget $LMDB_URL -O $LMDB_FILE
tar -C $TOOLS_DIR -xzvf $LMDB_FILE
cd $TOOLS_DIR/lmdb*/libraries/liblmdb/

# Using # as delimiter in the next line. Replacing the prefix path to CONDA_DIR
sed -i '29s#.*#prefix='"$CONDA_DIR"'#' Makefile
$MAKE
$MAKE install
popd
rm -f $LMDB_FILE
cd $A

echo "Installed LMDB"

CAFFE_URL="https://github.com/BVLC/caffe.git"
CAFFE_DIR="$TOOLS_DIR/caffe"

# Get source
git clone $CAFFE_URL $CAFFE_DIR

echo "Cloned Caffe"

# Install Protobuf & OpenCV library
conda install -c https://conda.anaconda.org/anaconda protobuf
conda install -c https://conda.anaconda.org/menpo opencv

#Install tensorflow
pip uninstall tensorflow
conda install -c conda-forge tensorflow


echo "Installed Protobuf, Conda through Conda and tensorflow"
