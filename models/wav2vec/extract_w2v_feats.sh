#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
mkdir -p checkpoints
W2V_MODELF=checkpoints/wav2vec_large.pt # Paths to wav2vec model files.

if [ ! -f "$W2V_MODELF" ]; then
    echo "Downloading wav2vec model checkpoints.."
    curl -#o $W2V_MODELF https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
fi

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting features using wav2vec-large model for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    ../../bin/gen_wav2vec_feats.py \
	    --use_gpu $W2V_MODELF $feats_dir $wav_dir/*.wav
done

echo
echo "Generating configuration files..."
../../bin/gen_config_files.py $FEATS_DIR configs/tasks/ \
    $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
