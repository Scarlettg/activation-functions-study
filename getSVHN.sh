#!/bin/bash

TRAIN_LINK="http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
TEST_LINK="http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

data_dir="datasets/"

cd $data_dir
curl -OL $TRAIN_LINK
curl -OL $TEST_LINK
