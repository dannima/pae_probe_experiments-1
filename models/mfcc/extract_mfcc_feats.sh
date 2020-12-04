#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
NJOBS=20
WL=0.035 # Window length in seconds.
STEP=0.01 # Step in seconds.

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
	data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting mfcc for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    ../../bin/gen_librosa_feats.py \
        --ftype	mfcc --config configs/feats/mfcc.yaml \
        --step $STEP --wl $WL --n_jobs $NJOBS \
        $feats_dir ${wav_dir}/*.wav
done

echo
echo "Generating configuration files..."
../../bin/gen_config_files.py --context_size 5 $FEATS_DIR configs/tasks/ \
    $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
