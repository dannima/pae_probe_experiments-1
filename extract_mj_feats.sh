#!/bin/bash

# Load virtual environment containing latest version of mockingjay.
source ../mockingjay/bin/activate

# Paths to mockingjay model files.
MJ_MODELF=states-500000.ckpt

for corpus in ctimit_final  ffmtimit_final  ntimit_final  stctimit_final  timit_final  wtimit_final; do
    wav_dir=/data/corpora/processed_timit_variants/${corpus}/wav

    echo "Extracting features using mockingjay model for ${corpus}..."
    feats_dir=feats/${corpus}/mockingjay
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    python bin/gen_mockingjay_feats.py \
	   --use_gpu $MJ_MODELF $feats_dir $wav_dir/*.wav
done

# Deactivate virtual environment.
deactivate