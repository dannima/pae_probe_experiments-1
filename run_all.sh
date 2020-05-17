#!/bin/bash
NJOBS=20

# Download models/install sw.

# Extract features.
./extract_w2v_feats.sh
./extract_librosa_feats.sh

# Run probe experiments.
bin/run_probe_exp.py \
    --n_jobs $NJOBS --device cuda:1 configs/tasks/sad_mfcc.yaml \
    > logs/sad_mfcc.stdout \
    2> logs/sad_mfcc.stderr &
bin/run_probe_exp.py \
    --n_jobs $NJOBS --device cuda:2 configs/tasks/sad_mel_fbank.yaml \
    > logs/sad_mel_fbank.stdout \
    2> logs/sad_mel_fbank.stderr &
bin/run_probe_exp.py \
    --n_jobs $NJOBS --device cuda:3 configs/tasks/sad_vqw2v.yaml \
    > logs/sad_vqw2v.stdout \
    2> logs/sad_vqw2v.stderr &
