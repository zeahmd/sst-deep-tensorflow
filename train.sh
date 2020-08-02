#!/bin/bash

#environment variables
MODELS=("rnn" "lstm" "bilstm" "gru" "conv1d")
PATIENCE=3

#variables
patience=$PATIENCE

#It will train total 20 models 4 for net.
for model in "${MODELS[@]}"; do
    set -x
    #run all and fine
        python run.py \
        --name ${model} \
        --patience ${patience} \
        --save
    #run root and fine
        python run.py \
        --name ${model} \
        --root \
        --patience ${patience} \
        --save
    #run all and binary
        python run.py \
        --name ${model} \
        --binary \
        --patience ${patience} \
        --save
    #run root and binary
        python run.py \
        --name ${model} \
        --binary \
        --root \
        --patience ${patience} \
        --save
    set +x
done