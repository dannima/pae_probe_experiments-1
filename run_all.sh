#!/bin/bash
NJOBS=20
CUDA_NUM=0

# Download models/install sw.

# Extract features.
./extract_w2v_feats.sh
./extract_librosa_feats.sh
./extract_mj_feats.sh
./extract_decoar_feats.sh

# Run probe experiments.
mkdir logs
for TASK in 'sad' 'syllable' 'sonorant' 'fricative'
do
	for MODEL in 'svm' 'lr' 'nn'
	do
		for REP in 'mfcc' 'mel_fbank' 'w2v_large' 'vqw2v' 'mj' 'decoar'
		do
			if (("$CUDA_NUM" > 5))
			then
				CUDA_NUM=0
			fi

			bin/run_probe_exp.py --n_jobs $NJOBS --device cuda:$CUDA_NUM \
			    configs/tasks/${TASK}_${MODEL}_${REP}.yaml \
			    > logs/${TASK}_${MODEL}_${REP}.stdout \
			    2> logs/${TASK}_${MODEL}_${REP}.stderr &

			(( CUDA_NUM += 1))
		done
	done

	bin/run_majority_vote.py -t $TASK > logs/majority_${TASK}.stdout \
	    2> logs/majority_${TASK}.stderr &
done

for TASK in 'phone'
do
	for MODEL in 'svm' 'lr' 'nn'
	do
		for REP in 'mfcc' 'mel_fbank' 'w2v_large' 'vqw2v' 'mj' 'decoar'
		do
			if (("$CUDA_NUM" > 5))
			then
				CUDA_NUM=0
			fi

			bin/run_phone_classification.py --n_jobs $NJOBS --device cuda:$CUDA_NUM \
			    configs/tasks/${TASK}_${MODEL}_${REP}.yaml \
			    > logs/${TASK}_${MODEL}_${REP}.stdout \
			    2> logs/${TASK}_${MODEL}_${REP}.stderr &

			(( CUDA_NUM += 1))
		done
	done

	bin/run_majority_vote.py -t $TASK > logs/majority_${TASK}.stdout \
	    2> logs/majority_${TASK}.stderr &
done
