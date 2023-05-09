"""
Sanitize the labels for each example.

Here are the rules:

(a) For every example that Amanpreet annotated, verify the label and map to the correct label space.
We want to do this because the labels had the same definition as before but were called different.
(b) For every example that Anna and Mackenzie, adjudicate the label that they disagree.
"""

import argparse
import pandas as pd
import os

def readArgs ():
    parser = argparse.ArgumentParser (description="Sanitize the labels of every example")
    parser.add_argument ('--v1-annotations-file', type=str, required=True, help="File contains examples that were annotated in v1")
    parser.add_argument ('--v2-annotations-file', type=str, required=True, help="File contains examples that were annotated in v2")
    parser.add_argument ('--amanpreet-labels-file', type=str, required=True, help="File contains amanpreet's labels for examples in v1")
    parser.add_argument ('--anna-labels-file', type=str, required=True, help="File contains anna's labels for examples in v2")
    parser.add_argument ('--mackenzie-labels-file', type=str, required=True, help="File contains mackenzie's labels for examples in v2")
    parser.add_argument ('--input-file', type=str, required=True, help="File the examples that are shuffled")
    parser.add_argument ('--output-file', type=str, required=True, help="File contains the sanitized output")
    parser.add_argument ('--start', type=int, required=False, default=0, help="Start index")
    parser.add_argument ('--end', type=int, required=False, default=250, help="End index")
    return parser.parse_args ()

def set_v1_prompt (example_item, label_item):
    # Show the annotation example and its ID
    print (f"ID:{example_item['ID'].values[0]}")
    print (f"Text:{example_item['context_100'].values[0]}")
    print (f"Spatial Relationship: {label_item['spatial_relation'].values[0]}")
    print (f"Temporal Span: {label_item['temporal_span'].values[0]}")
    print (f"Narrative Tense: {label_item['narrative_tense'].values[0]}")

    # Prompt the user to change the value of the 'b' column.
    change = input("Change annotations (Y/N)? ")

    # If the user wants to change the value, prompt them for the new value.
    if change == "Y" or change == "y":
        new_spatial = input("Enter new value for column 'spatial_relation': ")
        new_temporal = input("Enter new value for column 'temporal_span': ")
        new_narrative = input ("Enter new value for column 'narrative_tense': ")

        annotation = [example_item['ID'].values[0], 
                      "VALID" if label_item['spatial_relation'].values[0] not in ["BAD LOC", "BAD PER"] else "INVALID", 
                      new_spatial,
                      new_temporal,
                      new_narrative]
        print (f"Updated annotations: {annotation}")
        return annotation
    annotation = [example_item['ID'].values[0], 
                  "VALID" if label_item['spatial_relation'].values[0] not in ["BAD LOC", "BAD PER"] else "INVALID", 
                  label_item['spatial_relation'].values[0],
                  label_item['temporal_span'].values[0],
                  label_item['narrative_tense'].values[0]]
    print (f"Selected annotations: {annotation}")
    return annotation

def set_v2_prompt (example_item, label1_item, label2_item):
    items = [label1_item, label2_item]
    # Show the annotation example and its ID
    print (f"ID:{example_item['ID'].values[0]}")
    print (f"Text:{example_item['context_100'].values[0]}")
    print (f"Valid Relationship: {label1_item['valid_relation'].values[0]}, {label2_item['valid_relation'].values[0]}")
    print (f"Spatial Relationship: {label1_item['spatial_relation'].values[0]}, {label2_item['spatial_relation'].values[0]}")
    print (f"Temporal Span: {label1_item['temporal_span'].values[0]}, {label2_item['temporal_span'].values[0]}")
    print (f"Narrative Tense: {label1_item['narrative_tense'].values[0]}, {label2_item['narrative_tense'].values[0]}")

    # Prompt the user to change the value of the 'b' column.
    change = input("Change annotations (Y/N)? ")

    # If the user wants to change the value, prompt them for the new value.
    if change == "Y" or change == "y":
        new_valid = input ("Enter new value for column 'valid_relation': ")
        new_spatial = input("Enter new value for column 'spatial_relation': ")
        new_temporal = input("Enter new value for column 'temporal_span': ")
        new_narrative = input ("Enter new value for column 'narrative_tense': ")

        annotation = [example_item['ID'].values[0], 
                      new_valid, 
                      new_spatial,
                      new_temporal,
                      new_narrative]
        print (f"Updated annotations: {annotation}")
        return annotation
    else:
        anno_index = input("Choose annotator (0/1): ")
        annotation = [example_item['ID'].values[0], 
                      items[int(anno_index)]['valid_relation'].values[0].upper(),
                      items[int(anno_index)]['spatial_relation'].values[0],
                      items[int(anno_index)]['temporal_span'].values[0],
                      items[int(anno_index)]['narrative_tense'].values[0]]
        print (f"Selected annotations: {annotation}")
        return annotation

def main (args):
    # Read the examples from both the rounds
    v1_examples = pd.read_csv (args.v1_annotations_file, sep='\t')
    v2_examples = pd.read_csv (args.v2_annotations_file, sep='\t')

    # Read the labels from both the rounds
    amanpreeet_examples = pd.read_csv (args.amanpreet_labels_file, sep='\t')
    anna_examples = pd.read_csv (args.anna_labels_file, sep='\t', on_bad_lines='skip')
    mackenzie_examples = pd.read_csv (args.mackenzie_labels_file, sep='\t', on_bad_lines='skip')

    # Read the input file
    input_examples = pd.read_csv (args.input_file, sep='\t')

    # Read the output file
    if os.path.exists (args.output_file):
        output_examples = pd.read_csv (args.output_file, sep='\t')
        output_ids = output_examples["ID"].values
    else:
        output_ids = list ()

    answers = list ()
    for i, row in input_examples.loc[args.start:args.end].iterrows ():
        if row['ID'] in output_ids:
            continue
        if row['ID'] in v1_examples.ID.values:
            id = row['ID']
            ans = set_v1_prompt (v1_examples.query ('ID == @id'), amanpreeet_examples.query ('ID == @id'))
            answers.append (ans)
        elif row['ID'] in v2_examples.ID.values:
            id = row['ID']
            ans = set_v2_prompt (v2_examples.query ('ID == @id'), anna_examples.query ('ID == @id'), mackenzie_examples.query ('ID == @id'))
            answers.append (ans)
        else:
            # this should never happen
            print ("Something went wrong")
    
    answers_df = pd.DataFrame (answers, columns=["ID", "valid_relation", "spatial_relation", "temporal_span", "narrative_tense"])
    if not os.path.isfile (args.output_file):
        answers_df.to_csv (args.output_file, sep='\t', header=True, index=False)
    else:
        answers_df.to_csv (args.output_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    main (readArgs ())