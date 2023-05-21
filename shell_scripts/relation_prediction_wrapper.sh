#! /bin/bash
for context_size in context_50; do
    for task_name in validity spatial temporal_span narrative_tense; do
        /bin/bash relation_prediction.sh $task_name $context_size 1754 0;
        /bin/bash relation_prediction.sh $task_name $context_size 1754 1;
    done
done

