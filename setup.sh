#!/bin/bash

set -x
mkdir tensorboard_logs
mkdir trained
mkdir -p sst/embedding
cd sst/embedding
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
rm crawl-300d-2M.vec.zip
set +x