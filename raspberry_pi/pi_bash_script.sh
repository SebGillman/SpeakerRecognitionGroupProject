#!/bin/sh

# Update OS and get python 3.7/idle3
sudo apt-get update
sudo apt-get install python3.7 idle3

# install tflite_runtime package
pip3 install /home/pi/Downloads/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl

#####################################################################################
# Builds ArmNN and installs PyARMNN

# Increase virtual memory swapfile allocation
sudo vi /etc/dphys-swapfile
# Find the following line:
#    CONF_SWAPSIZE=100
# Change this line to:
#    CONF_SWAPSIZE=1024
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

# Install SCONS and CMAKE
sudo apt-get update
sudo apt-get install scons
sudo apt-get install cmake
mkdir armnn-tflite && cd armnn-tflite
export BASEDIR=`pwd`
git clone https://github.com/Arm-software/ComputeLibrary.git
git clone https://github.com/Arm-software/armnn
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
tar xf boost_1_64_0.tar.bz2
git clone -b v3.5.0 https://github.com/google/protobuf.git
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b
git clone https://github.com/google/flatbuffers.git
cd $BASEDIR/ComputeLibrary
scons extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 neon=1
cd $BASEDIR/boost_1_64_0
./bootstrap.sh
./b2 --build-dir=$BASEDIR/boost_1_64_0/build toolset=gcc link=static cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options install --prefix=$BASEDIR/boost
cd $BASEDIR/protobuf
git submodule update --init --recursive
sudo apt-get install autoconf
sudo apt-get install libtool
./autogen.sh
./configure --prefix=$BASEDIR/protobuf-host
make
make install
cd $BASEDIR/tensorflow
../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../protobuf-host
cd $BASEDIR
git clone https://github.com/google/flatbuffers.git
cd $BASEDIR/flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make
#Install SWIG
sudo apt-get install libpcre3 libpcre3-dev
cd $BASEDIR
mkdir swig
cd swig
wget http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz
chmod 777 swig-4.0.2.tar.gz
tar -xzvf swig-4.0.2.tar.gz
cd swig-4.0.2/
./configure --prefix=/home/pi/armnn-tflite/swigtool/
sudo make
sudo make install
sudo vi /etc/profile
# Add the following lines to /etc/profile
#   export SWIG_PATH=/home/pi/armnn-tflite/swigtool/bin
#   export PATH=$SWIG_PATH:$PATH
source /etc/profile
# Build Arm NN
cd $BASEDIR/armnn
mkdir build
cd build
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build -DBOOST_ROOT=$BASEDIR/boost -DTF_GENERATED_SOURCES=$BASEDIR/tensorflow-protobuf -DPROTOBUF_ROOT=$BASEDIR/protobuf-host -DBUILD_TF_LITE_PARSER=1 -DTF_LITE_GENERATED_PATH=$BASEDIR/tensorflow/tensorflow/lite/schema -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers -DFLATBUFFERS_LIBRARY=$BASEDIR/flatbuffers/libflatbuffers.a -DSAMPLE_DYNAMIC_BACKEND=1 -DDYNAMIC_BACKEND_PATHS=$BASEDIR/armnn/src/dynamic/sample -DARMCOMPUTENEON=1 -DBUILD_TF_PARSER=1
make
cp $BASEDIR/armnn/build/*.so $BASEDIR/armnn/
cd /home/pi/armnn-tflite/armnn/src/dynamic/sample
mkdir build
cd build
cmake -DBOOST_ROOT=$BASEDIR/boost  -DBoost_SYSTEM_LIBRARY=$BASEDIR/boost/lib/libboost_system.a -DBoost_FILESYSTEM_LIBRARY=$BASEDIR/boost/lib/libboost_filesystem.a -DARMNN_PATH=$BASEDIR/armnn/libarmnn.so ..
make

# Install PYARMNN
# Following instructions for "Standalone build" from:
# https://git.mlplatform.org/ml/armnn.git/tree/python/pyarmnn/README.md
export SWIG_EXECUTABLE=$BASEDIR/swigtool/bin/swig
export ARMNN_INCLUDE=$BASEDIR/armnn/include/
export ARMNN_LIB=$BASEDIR/armnn/build/
cd $BASEDIR/armnn/python/pyarmnn
sudo apt-get install python3.6-dev build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
python3 setup.py clean --all
python3 swig_generate.py -v
python3 setup.py build_ext --inplace
python3 setup.py sdist
python3 setup.py bdist_wheel
pip3 install dist/pyarmnn-21.0.0-cp37-cp37m-linux_armv7l.whl
sudo pip3 install opencv-python==3.4.6.27
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4
sudo apt-get install libqt4-test

