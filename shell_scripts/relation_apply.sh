#! /bin/bash

python ../scripts/relation_apply.py --validity-model-path ..data/models/bert_models/context_100/0/1754/validity.pt \
                                    --book-ids $(head -n 5 ../data/experiments/testsets/contemporary_litbank/contemporary_litbank.ids) \
                                    --collocation-dirs ../data/experiments/testsets/contemporary_litbank/collocations \
                                    --output-dir ../data/experiments/testsets/contemporary_litbank/predictions