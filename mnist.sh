#!/bin/bash

declare -a functions=("relu" "lrelu" "tanh" "elu" "swish" "new")

for func in "${functions[@]}" 
do
	time python MNIST3layers.py --activation "$func" >> ~/MNIST_logs/MNIST_3_"$func"
	time python MNIST12layers.py --activation "$func" >> ~/MNIST_logs/MNIST_12_"$func"
done


find . -maxdepth 1 -type f -print0 | tar -czvf MNIST.tar.gz --null -T -

exit 0
