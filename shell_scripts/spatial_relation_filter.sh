#! /bin/bash

DATA_DIR=/mnt/data0/ssoni/projects/mobility-books/data
mkdir -p $DATA_DIR/experiments/bad_categories
for training_size in 800 1600; do
	for context_size in 10 50 100; do
		# generate a random string
		random_prefix=$(openssl rand -hex 3)
		# append in a file
		printf "%d\t%d\t%s\n" $training_size $context_size $random_prefix >> $DATA_DIR/experiments/bad_categories/metadata.tsv
		# call the prediction script
		python ../scripts/spatial_relation_filter.py --input-filename $DATA_DIR/annotations/final_annotations/final_annotations.v1.tsv \
								 --num-training-examples $training_size \
								 --num-epochs 15 \
								 --context context_$context_size \
								 --output-filename $DATA_DIR/experiments/bad_categories/$random_prefix.tsv
	done
done