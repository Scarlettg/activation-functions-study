#!/bin/bash

cifar100_link="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
cifar100_name="cifar-100-python.tar.gz"
data_dir="datasets"

curl -OL $cifar100_link && tar -xzvf $cifar100_name -C $data_dir

echo 0
