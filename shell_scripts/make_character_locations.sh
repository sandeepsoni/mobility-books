#! /bin/bash

INPUT_DIR=$1    # The BookNLP directory.
OUTPUT_DIR=$2   # The Collocations directory.
IDS_FILE=$3     # IDS in a single file

#parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename {} --output-dir ../data/v2 --window-size 10 ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_1_4/*.entities)
#parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename {} --output-dir ../data/v2 --window-size 10 ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_5_9/*.entities)

parallel --jobs 16 python ../scripts/name_location_cooccurrences.py --input-filename $INPUT_DIR/{}.entities --output-dir $OUTPUT_DIR --window-size 10 :::: $IDS_FILE