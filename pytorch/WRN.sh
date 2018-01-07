
#!/bin/bash

save=$1
depth=$2
width=$3
batchSize=$4
dataset=$5
dataroot=$6
activation=$7





if [ "$activation" == "all" ];
	then 
		declare -a functions=("relu" "lrelu" "tanh" "elu" "swish" "new")
	else
		declare -a functions=("$activation")
fi

#declare -a functions=("relu" "lrelu" "tanh" "elu" "swish")
for func in "${functions[@]}" 
do
	modelpath=$1"/""$5""_WRN-""$2""-""$3""_batch_size-""$4""_activation-""$func"

	echo "Running--------------" python main.py --save $modelpath --depth "$depth" \
	 --width "$width" --dataset "$dataset" --dataroot "$dataroot" --activation "$func" \
	 --batchSize "$batchSize" --cuda

	time python main.py --save $modelpath --depth "$depth" \
	 --width "$width" --dataset "$dataset" --dataroot "$dataroot" --activation "$func" \
	 --batchSize "$batchSize" --cuda
done


# name="cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4""_activation-*"
# name2="cifar10_resnet-""$2""_train_epochs-""$3""_batch_size-""$4"
# cd $1
# find . -maxdepth 1 -type d -name "$name"  -print0 | tar -czvf "$name2".tar.gz --null -T -

exit 0
