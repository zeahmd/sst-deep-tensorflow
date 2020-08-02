#!/bin/bash

if [ -d "trained" ]; then
    for dir in trained/*; do
        for file in $dir/*.{h5,}; do
            if [ -f $file ]; then
                set -x
                    python run.py \
                    --test \
                    --filename $file
                set +x
            fi
        done
    done
else
    echo "trained directory don't exist!"
fi
