python ../scripts/valid_relation_train_and_eval.py \
       --pretrained-model-name bert-base-cased \
       --training-data-file ../data/annotations/final_annotations/final_annotations.v1.pickle \
       --num-epochs 15 \
       --context-field context_100 \
       --model-path ../data/experiments/bad_categories/model_checkpoints/trained.v1.pt \
       --num-labels 2 \
       --test-ids-file ../data/annotations/final_annotations/common_ids.txt