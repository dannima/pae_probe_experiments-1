#!/usr/bin/env bash

FEATS_DIR=feats # Parent directory of output features.
mkdir -p checkpoints

# Paths to wav2vec 2.0 model files.
W2V2_MODELF=checkpoints/wav2vec2_vox_960h_new.pt
LETTER_DICT=checkpoints/dict.ltr.txt
VOCAB_DIR="$(dirname "$LETTER_DICT")"

if [ ! -f "$W2V2_MODELF" ]; then
    echo "Downloading wav2vec 2.0 model checkpoints.."
    curl -#o $W2V2_MODELF https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h_new.pt
fi

if [ ! -f "$LETTER_DICT" ]; then
    echo "Downloading the language model vocabulary used in wav2vec 2.0 (wav2letter).."
    curl -#o $LETTER_DICT https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
fi

echo 'Update fairseq to its newest version'
echo 'A temporary workaround for loading wav2vec 2.0..'
pip install soundfile git+git://github.com/pytorch/fairseq.git@66e1803c60272602c719a5ba75acef1c530066ef
# pip install soundfile git+git://github.com/pytorch/fairseq.git@master
# Successfully uninstalled fairseq-1.0.0a0+3aeb8fe

for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    data_dir="${corpus^^}_PATH"
    wav_dir=${!data_dir}/wav

    echo "Extracting features using wav2vec 2.0 large model for ${corpus}..."
    feats_dir=$FEATS_DIR/${corpus}
    export CUDA_VISIBLE_DEVICES=`free-gpu`
    ../../bin/gen_wav2vec_feats.py \
        --use_gpu --v2 --vocab $VOCAB_DIR $W2V2_MODELF \
        $feats_dir $wav_dir/*.wav
done

# echo
# echo "Generating configuration files..."
# ../../bin/gen_config_files.py --step 0.02 $FEATS_DIR configs/tasks/ \
#     $TIMIT_PATH $NTIMIT_PATH $CTIMIT_PATH $FFMTIMIT_PATH $STCTIMIT_PATH $WTIMIT_PATH
