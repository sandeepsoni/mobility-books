"""
python format_v1_annotations.py \
    --annotation-file ../data/annotations/amanpreet.v1.tsv \
    --sample-file ../data/annotations/sample_annotations.v1.modified.tsv \
    --output-file ../data/annotations/examples_to_annotate.v1.tsv
"""
import argparse
import pandas as pd
import numpy as np

def readArgs ():
    parser = argparse.ArgumentParser (description="Format the annotations in the V1 annotation round")
    parser.add_argument ("--annotation-file", required=True, type=str, help="The annotation files from the V1 round")
    parser.add_argument ("--sample-file", required=True, type=str, help="The sampled file from which the annotations were done")
    parser.add_argument ("--output-file", required=True, type=str, help="The output annotation file")
    return parser.parse_args ()

def insert_tags (row, 
                 text_field, 
                 persons_start_index,
                 persons_end_index,
                 locations_start_index,
                 locations_end_index, 
                 offset_from_start):
    tokens = row[text_field].split ()
    start_person = row[persons_start_index] - row[offset_from_start]
    end_person = (row[persons_end_index] - row[offset_from_start])+1 # because this tag will be inserted after
    start_loc = row[locations_start_index] - row[offset_from_start]
    end_loc = row[locations_end_index] - row[offset_from_start]+1 # because this tag will be inserted after
    tags = np.array(['<char>', '</char>', '<place>', '</place>'])
    indices = np.array([start_person, end_person, start_loc, end_loc])

    ind = np.argsort (-np.array (indices)) # sort in descending order and insert one by one
    sorted_indices = indices[ind]
    sorted_tags = tags[ind]

    for index, tag in zip (sorted_indices, sorted_tags):
        tokens.insert (index, tag)

    return ' '.join (tokens)


def get_span (row, text_field, start_index, end_index, offset_from_start):
    tokens = row[text_field].split ()
    start = row[start_index] - row[offset_from_start]
    end = row[end_index] - row[offset_from_start]
    return ' '.join (tokens[start:end+1])

def main (args):

    # Read the sample file from which the annotations are sourced
    sampled_df = pd.read_csv (args.sample_file, sep='\t')
    # drop needless columns
    sampled_df.drop (labels=["is_character_correct", "is_location_correct", "is_character_at_location"], axis=1, inplace=True)

    # Read the annotations file
    annotations_df = pd.read_csv (args.annotation_file, sep='\t')
    # Drop the duplicate IDs
    annotations_df = annotations_df.drop_duplicates ("ID")

    # Merge the two dataframes
    merged_df = pd.merge (left=sampled_df, right=annotations_df, how="inner", on="ID", suffixes=('', '_y'))
    merged_df.drop(merged_df.filter(regex='_y$').columns, axis=1, inplace=True) # drop the columns that have been repeated
    
    for offset in [10, 50, 100]:
        merged_df[f'offset_{offset}'] = merged_df[['persons_start_token', 'locations_start_token']].min (axis=1) - offset
    
    merged_df['persons_text'] = merged_df.apply (lambda x: get_span (x, 
                                                                     'context_10', 
                                                                     'persons_start_token', 
                                                                     'persons_end_token', 
                                                                     'offset_10'), 
                                                            axis=1)
    merged_df['locations_text'] = merged_df.apply (lambda x: get_span (x,
                                                                       'context_10',
                                                                       'locations_start_token',
                                                                       'locations_end_token',
                                                                       'offset_10'),
                                                            axis=1)
    merged_df['modified_context_10'] = merged_df.apply (lambda x: insert_tags (x,
                                                                               'context_10',
                                                                               'persons_start_token',
                                                                               'persons_end_token',
                                                                               'locations_start_token',
                                                                               'locations_end_token',
                                                                               'offset_10'), 
                                                        axis=1)
    
    merged_df['modified_context_50'] = merged_df.apply (lambda x: insert_tags (x,
                                                                               'context_50',
                                                                               'persons_start_token',
                                                                               'persons_end_token',
                                                                               'locations_start_token',
                                                                               'locations_end_token',
                                                                               'offset_50'), 
                                                        axis=1)
    
    merged_df['modified_context_100'] = merged_df.apply (lambda x: insert_tags (x,
                                                                               'context_100',
                                                                               'persons_start_token',
                                                                               'persons_end_token',
                                                                               'locations_start_token',
                                                                               'locations_end_token',
                                                                               'offset_100'), 
                                                        axis=1)
    
    merged_df = merged_df[['ID',
                           'book_id',
                           'persons_text',
                           'locations_text',
                           'modified_context_10',
                           'modified_context_50',
                           'modified_context_100']]
    
    merged_df = merged_df.rename(columns={"modified_context_10": "context_10", 
                                          "modified_context_50": "context_50", 
                                          "modified_context_100": "context_100",
                                          "persons_text": "char_text",
                                          "locations_text": "place_text"})
    
    merged_df.to_csv (args.output_file, sep='\t', header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())