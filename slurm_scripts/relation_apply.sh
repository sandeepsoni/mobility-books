#!/bin/bash

# Job name: Give your job a meaningful name

#SBATCH --job-name=temporal-models-finetuning

# Partition: choose the correct machine, and below you can specify the number of nodes and tasks. 
# This is a magical mystery resolved by reading savio documentation or consulting with their help desk, e.g. https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/

#SBATCH --partition=savio3_gpu 


# All the other options

#SBATCH --time=4:00:00

#SBATCH --account=fc_dbamman

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:GTX2080TI:1

#SBATCH --qos=gtx2080_gpu3_normal

#SBATCH --mail-user=sandeepsoni@berkeley.edu

#SBATCH --mail-type=all

HOME_DIR=/global/home/users/sandeepsoni
USERS_DIR=/global/scratch/users/sandeepsoni
PROJECTS_DIR=$HOME_DIR/projects/mobility-books
SCRATCH_DIR=$USERS_DIR/projects/mobility-books

conda activate nlp-urap
export TRANSFORMERS_CACHE="/global/scratch/users/sandeepsoni/models/transformers" # Keep all the models in this directory
export HF_DATASETS_CACHE="/global/scratch/users/sandeepsoni/datasets/transformers" # Keep all the datasets in this directory

cd $PROJECTS_DIR/scripts
module load cuda/10.2
hostname

python relation_apply.py --validity-model-path $SCRATCH_DIR/models/bert_models/context_100/0/1754/validity.pt \
                                    --spatial-model-path $SCRATCH_DIR/models/bert_models/context_100/0/1754/spatial.pt \
                                    --temporal-span-model-path $SCRATCH_DIR/models/bert_models/context_100/0/1754/temporal_span.pt \
                                    --narrative-tense-model-path $SCRATCH_DIR/models/bert_models/context_100/0/1754/narrative_tense.pt \
                                    --book-ids $(cat $IDS_FILE) \
                                    --collocations-dir $SCRATCH_DIR/experiments/testsets/contemporary_litbank/collocations \
                                    --output-dir $SCRATCH_DIR/experiments/testsets/contemporary_litbank/predictions

conda deactivate
