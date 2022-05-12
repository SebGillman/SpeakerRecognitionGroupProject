#!/bin/sh

# Update OS and get python 3.7/idle3
sudo apt-get update
sudo apt-get install python3.7 idle3

# install tflite_runtime package
pip3 install /home/pi/Downloads/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl

### ASSUMPTIONS ###
# Folder exists: /home/pi/TFLite_Group
# Folder contains:
# group_model.tflite
# group_labels.txt
# adam_test.wav

