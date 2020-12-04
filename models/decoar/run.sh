#!/usr/bin/env bash
WORKING_DIR=$(pwd)

cd ../../
. path.sh
cd $WORKING_DIR
./extract_decoar_feats.sh
# ./../../run_probe_exp.sh