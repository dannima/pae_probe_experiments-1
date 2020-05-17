#!/bin/bash

# Load virtual environment containing latest version of fairseq.
source ../venv/bin/activate

# Paths to wav2vec/vq-wav2vec model files.
W2V_MODELF=wav2vec/models/wav2vec_large.pt
W2V_FEATS_DIR=feats/wav2vec-large
VQW2V_MODELF=wav2vec/models/vq-wav2vec_kmeans.pt
VQW2V_FEATS_DIR=feats/vq-wav2vec_kmeans_roberta
ROBERTA_MODELF=wav2vec/models/roberta/bert_kmeans.pt
ROBERTA_VOCABF=wav2vec/models/roberta/dict.txt

for corpus in ctimit_final  ffmtimit_final  ntimit_final  stctimit_final  timit_final  wtimit_final; do
    wav_dir=/data/corpora/processed_timit_variants/${corpus}/wav

    echo "Extracting features using wav2vec-large model for ${corpus}..."
    feats_dir=feats/${corpus}/wav2vec-large
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    python bin/gen_wav2vec_feats.py \
	   --use_gpu $W2V_MODELF $feats_dir $wav_dir/*.wav

    echo "Extracting features using vq-wav2vec-kmeans and RoBERTa model for ${corpus}..."
    feats_dir=feats/${corpus}/vq-wav2vec_kmeans_roberta
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    python bin/gen_wav2vec_feats.py \
	   --use_gpu --vocab $ROBERTA_VOCABF --roberta $ROBERTA_MODELF \
	   $VQW2V_MODELF $feats_dir $wav_dir/*.wav
done

# Deactivate virtual environment.
deactivate
