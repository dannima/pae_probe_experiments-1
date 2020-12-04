#!/usr/bin/env bash

# Change *TIMIT_PATH to the path that your dataset is stored.
export TIMIT_PATH='/data/corpora/processed_timit_variants/timit_final'
export NTIMIT_PATH='/data/corpora/processed_timit_variants/ntimit_final'
export CTIMIT_PATH='/data/corpora/processed_timit_variants/ctimit_final'
export FFMTIMIT_PATH='/data/corpora/processed_timit_variants/ffmtimit_final'
export STCTIMIT_PATH='/data/corpora/processed_timit_variants/stctimit_final'
export WTIMIT_PATH='/data/corpora/processed_timit_variants/wtimit_final'

FILE="phones.60-48-39.map"
if [ ! -f "$FILE" ]; then
	echo "Downloading 61-to-39 phone mapping..."
	curl -O https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/timit/s5/conf/phones.60-48-39.map
fi