#!/bin/bash

model_dir=$1
resnet_size=$2
train_epochs=$3
batch_size=$4
activation=$5



if [ "$activation" == "all" ];
	then 
		declare -a functions=("relu" "lrelu" "tanh" "elu" "swish")
	else
		declare -a functions=("$activation")
fi

#declare -a functions=("relu" "lrelu" "tanh" "elu" "swish")
for func in "${functions[@]}" 
do
	modelpath=$1"/cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4""_activation-""$func"
	logpath=$1"/log_cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4""_activation-""$func"

	echo "Running--------------" python cifar10_main.py --model_dir $modelpath --resnet_size "$resnet_size" \
	 --train_epochs "$train_epochs" --batch_size "$batch_size" --activation "$func"

	time python cifar10_main.py --model_dir $modelpath --resnet_size "$resnet_size" \
	 --train_epochs "$train_epochs" --batch_size "$batch_size" --activation "$func" >> "$logpath"
done


name="cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4""_activation-*"
name2="cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4"
cd $1
find . -maxdepth 1 -type d -name "$name"  -print0 | tar -czvf "$name2".tar.gz --null -T -

exit 0
