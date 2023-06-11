"""
Sanitize the labels for each example.

Here are the rules:

(a) For every example that Amanpreet annotated, verify the label and map to the correct label space.
We want to do this because the labels had the same definition as before but were called different.
(b) For every example that Anna and Mackenzie, adjudicate the label that they disagree.

Usage: 
Author: Sandeep Soni
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

def valid_prompt ():
    option = input ("Select option (INVALID=0, VALID=1): ")
    match option:
        case "0":
            return "INVALID"
        case "1":
            return "VALID"
        case _:
            return "INVALID"

def spatial_prompt ():
    option = input("Select option (IN=0,NEAR=1,THRU=2,TO=3,FROM=4,NO REL=5): ")
    match option:
        case "0":
            return "IN"
        case "1":
            return "NEAR"
        case "2":
            return "THRU"
        case "3":
            return "TO"
        case "4":
            return "FROM"
        case "5":
            return "NO REL"
        case _:
            return ""

def temporal_span_prompt ():
    option = input ("Select option (PUNCTUAL=0, HABITUAL=1): ")
    match option:
        case "0":
            return "PUNCTUAL"
        case "1":
            return "HABITUAL"
        case _:
            return ""

def narrative_tense_prompt ():
    option = input ("Select option (ONGOING=0, ALREADY HAPPENED=1): ")
    match option:
        case "0":
            return "ONGOING"
        case "1":
            return "ALREADY HAPPENED"
        case _:
            return ""

def set_v1_prompt (example_item, label_item):
    # Show the annotation example and its ID
    print (f"ID:{example_item['ID'].values[0]}")
    print (f"Text:{example_item['context_100'].values[0]}")
    print (f"Spatial Relationship: {label_item['spatial_relation'].values[0]}")
    print (f"Temporal Span: {label_item['temporal_span'].values[0]}")
    print (f"Narrative Tense: {label_item['narrative_tense'].values[0]}")

    spatial_map = {
        "IN": "IN",
        "FROM": "FROM",
        "THROUGH": "THRU",
        "NEAR": "NEAR",
        "NO RELATIONSHIP ASSERTED": "NO REL",
        "NEGATIVE ASSERTION": "NO REL",
        "TOWARD(got there)": "TO",
        "UNCERTAIN ASSERTION": "NO REL",
        "TOWARD (uncertain got there)": "NO REL"
    }

    # Prompt the user to change the value of the 'b' column.
    change = input("Change annotations (Y/N)? ")

    # If the user wants to change the value, prompt them for the new value.
    if change == "Y" or change == "y":
        new_valid = valid_prompt()
        new_spatial = spatial_prompt ()
        new_temporal = temporal_span_prompt ()
        new_narrative = narrative_tense_prompt ()

        annotation = [example_item['ID'].values[0], 
                      new_valid, 
                      new_spatial,
                      new_temporal,
                      new_narrative]
        print (f"Updated annotations: {annotation}")
        return annotation
    valid = "VALID" if label_item['spatial_relation'].values[0] not in ["BAD LOC", "BAD PER"] else "INVALID"
    annotation = [example_item['ID'].values[0], 
                  valid, 
                  spatial_map [label_item['spatial_relation'].values[0]] if valid == "VALID" else "",
                  label_item['temporal_span'].values[0] if valid == "VALID" else "",
                  label_item['narrative_tense'].values[0] if valid == "VALID" else ""]
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
        new_valid = valid_prompt()
        new_spatial = spatial_prompt ()
        new_temporal = temporal_span_prompt ()
        new_narrative = narrative_tense_prompt ()

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

    for i, row in input_examples.loc[args.start:args.end].iterrows ():
        answers = list ()
        if row['ID'] in output_ids:
            print (f"{row['ID']} already annotated")
            continue
        if row['ID'] in v1_examples.ID.values:
            id = row['ID']
            print (f"{row['ID']} annotating from v1")
            ans = set_v1_prompt (v1_examples.query ('ID == @id'), amanpreeet_examples.query ('ID == @id'))
            answers.append (ans)
        elif row['ID'] in v2_examples.ID.values:
            id = row['ID']
            print (f"{row['ID']} annotating from v2")
            # Only annotate if there is disagreement
            ans = set_v2_prompt (v2_examples.query ('ID == @id'), anna_examples.query ('ID == @id'), mackenzie_examples.query ('ID == @id'))
            answers.append (ans)
        else:
            # this should never happen
            print ("Something went wrong")
    
        answers_df = pd.DataFrame (answers, columns=["ID", "valid_relation", "spatial_relation", "temporal_span", "narrative_tense"])
        if not os.path.isfile (args.output_file):
            answers_df.to_csv (args.output_file, sep='\t', header=True, index=False)
        else:
            answers_df.to_csv (args.output_file, mode='a', sep='\t', header=False, index=False)

if __name__ == "__main__":
    main (readArgs ())