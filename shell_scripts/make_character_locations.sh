#! /bin/bash

OUTPUT_DIR=$1
IDS_FILE=$2

#parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename {} --output-dir ../data/v2 --window-size 10 ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_1_4/*.entities)
#parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename {} --output-dir ../data/v2 --window-size 10 ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_5_9/*.entities)

parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename {}.entities --output-dir $OUTPUT_DIR --window-size 10 :::: $IDS_FILE