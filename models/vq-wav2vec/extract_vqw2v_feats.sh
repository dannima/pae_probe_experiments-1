#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
mkdir -p checkpoints

# Paths to vq-wav2vec model files.
VQW2V_MODELF=checkpoints/vq-wav2vec_kmeans.pt
ROBERTA_MODELF=checkpoints/bert_kmeans.pt
ROBERTA_VOCABF=checkpoints/dict.txt

if [ ! -f "$VQW2V_MODELF" ]; then
    echo "Downloading vq-wav2vec model checkpoints..."
    curl -#o $VQW2V_MODELF https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
fi

if [ ! -f "$ROBERTA_MODELF" ] || [ ! -f "$ROBERTA_VOCABF" ]; then
    echo "Downloading RoBERTa model and vocabulary..."
    curl -#o checkpoints/bert_kmeans.tar https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar
    echo "Extracting..."
    cd checkpoints/
    tar -xvf bert_kmeans.tar
    rm bert_kmeans.tar
    cd ../
fi

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting features using vq-wav2vec-kmeans and RoBERTa model for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    ../../bin/gen_wav2vec_feats.py \
        --use_gpu --vocab $ROBERTA_VOCABF --roberta $ROBERTA_MODELF \
        $VQW2V_MODELF $feats_dir $wav_dir/*.wav

done

echo
echo "Generating configuration files..."
../../bin/gen_config_files.py $FEATS_DIR configs/tasks/ \
    $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
