#! /bin/bash

TASK_NAME=${1:-validity} #validity
CONTEXT_FIELD=${2:-context_10} #context_10
NUM_TRAINING_EXAMPLES=${3:-100} #100
NUM_HIDDEN=${4:-0} #0

python ../scripts/relation_prediction.py --task-name $TASK_NAME \
                                         --train-data-file ../data/annotations/train.tsv \
                                         --train-labels-file ../data/annotations/train.labels.tsv \
                                         --dev-data-file ../data/annotations/dev.tsv \
                                         --dev-labels-file ../data/annotations/dev.labels.tsv \
                                         --models-dir ../data/models/bert_models/$CONTEXT_FIELD/$NUM_HIDDEN/$NUM_TRAINING_EXAMPLES \
                                         --num-training-examples $NUM_TRAINING_EXAMPLES \
                                         --results-dir ../data/results/bert_results/$CONTEXT_FIELD/$NUM_HIDDEN/$NUM_TRAINING_EXAMPLES \
                                         --text-field $CONTEXT_FIELD \
                                         --num-epochs 15 \
                                         --num-hidden $NUM_HIDDEN