#! /bin/bash

python ../scripts/relation_apply.py --validity-model-path ../data/models/bert_models/context_100/0/1754/validity.pt \
                                    --spatial-model-path ../data/models/bert_models/context_100/0/1754/spatial.pt \
                                    --temporal-span-model-path ../data/models/bert_models/context_100/0/1754/temporal_span.pt \
                                    --narrative-tense-model-path ../data/models/bert_models/context_100/0/1754/narrative_tense.pt \
                                    --book-ids $(cat $1) \
                                    --collocations-dir ../data/experiments/testsets/contemporary_litbank/collocations \
                                    --output-dir ../data/experiments/testsets/contemporary_litbank/predictions
