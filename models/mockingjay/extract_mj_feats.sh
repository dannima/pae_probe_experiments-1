#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
MJ_MODELF=checkpoints/states-500000.ckpt # Paths to mockingjay model files.

if [ ! -f "$MJ_MODELF" ]; then
    echo "Mockingjay model not found!"
    echo "Please download Mockingjay from https://drive.google.com/drive/folders/1d7nFh2I0J8EGdXJJ2_7zIeTcYzPtiX1Y"
    exit 1
fi

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting features using Mockingjay model for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    ../../bin/gen_mockingjay_feats.py \
	   --use_gpu $MJ_MODELF $feats_dir $wav_dir/*.wav

done

echo
echo "Generating configuration files..."
../../bin/gen_config_files.py --step 0.0125 $FEATS_DIR configs/tasks/ \
    $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
