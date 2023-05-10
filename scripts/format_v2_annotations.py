"""
python format_v2_annotations.py \
    --annotation-files ../data/annotations/anna.v2.tsv ../data/annotations/mackenzie.v2.tsv \
    --output-file ../data/annotations/examples_to_annotate.v2.tsv
"""

import argparse
import pandas as pd
import numpy as np

def readArgs ():
    parser = argparse.ArgumentParser (description="Format the annotations in the V1 annotation round")
    parser.add_argument ("--annotation-files", required=True, type=str, nargs='+', help="The annotation files from the V2 round")
    parser.add_argument ("--output-file", required=True, type=str, help="The output annotation file")
    return parser.parse_args ()

def main (args):
    # Read the annotations file
    annotations_df = pd.concat([pd.read_csv (filename, sep='\t', on_bad_lines='skip')for filename in args.annotation_files])
    # Keep only those rows that have an annotation
    annotations_df = annotations_df.query('not valid_relation.isnull()')   
    annotations_df.drop (['valid_relation', 'spatial_relation', 'temporal_span', 'narrative_tense'], axis=1, inplace=True)
    annotations_df = annotations_df.drop_duplicates ("ID")
    annotations_df = annotations_df[['ID',
                                     'book_id',
                                     'char_text',
                                     'place_text',
                                     'context_10',
                                     'context_50',
                                     'context_100']]
    
    annotations_df.to_csv (args.output_file, sep='\t', header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())