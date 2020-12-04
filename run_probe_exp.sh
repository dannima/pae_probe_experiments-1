#!/usr/bin/env bash

CUDA_NUM=0
NJOBS=20
NUM_GPU=6 #Change NUM_GPU to number of GPUs you have.
echo "Run probing experiments..."
mkdir -p logs
for TASK in 'sad' 'vowel' 'sonorant' 'fricative'
do
	for MODEL in 'max_margin' 'logistic' 'nnet'
	do
		if [ "$CUDA_NUM" -ge "$NUM_GPU" ]
		then
			CUDA_NUM=0
		fi

		../../bin/run_probe_exp.py --n_jobs $NJOBS --device cuda:$CUDA_NUM \
		    configs/tasks/${TASK}_${MODEL}.yaml \
		    > logs/${TASK}_${MODEL}.stdout \
		    2> logs/${TASK}_${MODEL}.stderr &

		let "CUDA_NUM += 1"
	done
done

for TASK in 'phone'
do
	for MODEL in 'max_margin' 'logistic' 'nnet'
	do
		if [ "$CUDA_NUM" -ge "$NUM_GPU" ]
		then
			CUDA_NUM=0
		fi

		../../bin/run_phone_classification.py --n_jobs $NJOBS --device cuda:$CUDA_NUM \
		    configs/tasks/${TASK}_${MODEL}.yaml \
		    > logs/${TASK}_${MODEL}.stdout \
		    2> logs/${TASK}_${MODEL}.stderr &

		let "CUDA_NUM += 1"
	done
done
