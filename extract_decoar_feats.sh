#!/bin/bash

# Load virtual environment.
source ../mockingjay/bin/activate

# Paths to DeCoAR model files.
DECOAR_MODELF=../speech-representations/artifacts/decoar-encoder-29b8e2ac.params

for corpus in ctimit_final  ffmtimit_final  ntimit_final  stctimit_final  timit_final  wtimit_final; do
    wav_dir=/data/corpora/processed_timit_variants/${corpus}/wav

    echo "Extracting features using DeCoAR model for ${corpus}..."
    feats_dir=feats/${corpus}/decoar
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    python bin/gen_decoar_feats.py \
	   --use_gpu $DECOAR_MODELF $feats_dir $wav_dir/*.wav
done

# Deactivate virtual environment.
deactivate