#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
mkdir -p checkpoints
DECOAR_MODELF=checkpoints/decoar-encoder-29b8e2ac.params # Paths to DeCoAR model files.

if [ ! -f "$DECOAR_MODELF" ]; then
    echo "Downloading DeCoAR model checkpoints.."
    wget -qO- https://apache-mxnet.s3-us-west-2.amazonaws.com/gluon/models/decoar-encoder-29b8e2ac.zip | \
        zcat > checkpoints/decoar-encoder-29b8e2ac.params
fi

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting features using DeCoAR model for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    ../../bin/gen_decoar_feats.py \
	    --use_gpu $DECOAR_MODELF $feats_dir $wav_dir/*.wav
done

echo
echo "Generating configuration files..."
../../bin/gen_config_files.py $FEATS_DIR configs/tasks/ \
    $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
