#! /bin/bash

python ../scripts/create_annotation_sample.py --collocation-dir ../data/v2/ --content-dir /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_1_4/ /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_5_9/ --output-file ../data/annotations/david_bamman/sample_annotation.tsv --sample-size 10000 --context-window-sizes 10 50 100
