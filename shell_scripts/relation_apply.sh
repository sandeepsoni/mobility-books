#! /bin/bash

VALIDITY_MODEL=$1
SPATIAL_MODEL=$2
TEMPORAL_SPAN_MODEL=$3
NARRATIVE_TENSE_MODEL=$4

python ../scripts/relation_apply.py --validity-model-path $VALIDITY_MODEL \
                                    --spatial-model-path $SPATIAL_MODEL \
                                    --temporal-span-model-path $TEMPORAL_SPAN_MODEL \
                                    --narrative-tense-model-path $NARRATIVE_TENSE_MODEL \
                                    --book-ids-file ../data/experiments/testsets/conlit/conlit.ids \
                                    --examples-dir ../data/experiments/testsets/conlit/examples \
                                    --output-dir ../data/experiments/testsets/conlit/predictions
