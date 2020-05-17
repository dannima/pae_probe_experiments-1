#!/bin/bash
FEATS_DIR=feats # Parent directory of all output features.
NJOBS=20
WL=0.035 # Window length in seconds.
STEP=0.01 # Step in seconds.
for corpus in ctimit_final ffmtimit_final  ntimit_final  stctimit_final  timit_final  wtimit_final; do
    wav_dir=/data/corpora/processed_timit_variants/${corpus}/wav

    echo "Extracting MFCCs for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}/mfcc
    bin/gen_librosa_feats.py \
	--ftype mfcc --config configs/feats/mfcc.yaml \
	--step $STEP --wl $WL --n_jobs $NJOBS \
	$feats_dir ${wav_dir}/*.wav

    echo "Extracting mel filterbank for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}/mel_fbank
    bin/gen_librosa_feats.py \
        --ftype	mel_fbank --config configs/feats/mel_fbank.yaml \
        --step $STEP --wl $WL --n_jobs $NJOBS \
        $feats_dir ${wav_dir}/*.wav
done
