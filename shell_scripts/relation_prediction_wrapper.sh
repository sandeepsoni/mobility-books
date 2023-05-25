#! /bin/bash
for context_size in context_10 context_50 context_100; do
    for task_name in validity spatial temporal_span narrative_tense; do
        for num_training_examples in 1754; do
            for num_hidden in 0 1; do
                /bin/bash relation_prediction.sh $task_name $context_size $num_training_examples $num_hidden;
            done
        done
    done
done

