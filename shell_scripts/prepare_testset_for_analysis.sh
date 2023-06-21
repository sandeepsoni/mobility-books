#!/bin/bash

# Run on redwood
#parallel --jobs 16 \
#         python ../scripts/prepare_testset.py \
#         --book-id {} \
#         --collocations-dir ../data/v3 \
#         --content-dir /data0/ssoni/booknlp/contemporary_litbank \
#         --context-window-size 100 \
#         --output-dir ../data/experiments/testsets/contemporary_litbank/collocations \
#         ::: $(cat ../data/experiments/testsets/contemporary_litbank/contemporary_litbank.ids)

parallel --jobs 16 \
         python ../scripts/prepare_testset.py \
         --book-id {} \
         --collocations-dir ../data/experiments/testsets/conlit/collocations \
         --content-dir ../data/experiments/testsets/conlit/booknlp \
         --context-window-size 100 \
         --output-dir ../data/experiments/testsets/conlit/examples \
         :::: ../data/experiments/testsets/conlit/conlit.ids