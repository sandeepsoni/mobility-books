"""
Amanpreet's annotations did not have the sentence ends. 
So for all the examples in the sample that Amanpreet used, we'll try to get the sentence end token indices.
"""

import argparse
import os
import pandas as pd

def readArgs ():
    parser = argparse.ArgumentParser (description="Get the sentence ends for each example")
    parser.add_argument ('--dir-paths', type=str, required=True, nargs='+', help="Directory paths that contain bookNLP files")
    parser.add_argument ('--sample-file', type=str, required=True, help="File contains the sample")
    parser.add_argument ("--output-file", type=str, required=True, help="File contains the output")
    args = parser.parse_args ()
    return args

def correct_path (paths, filename):
    for path in paths:
        if os.path.exists (os.path.join (path, filename)) and os.path.isfile (os.path.join (path, filename)):
            return os.path.join (path, filename)
        
def read_file (filename, sep='\t'):
    with open (filename) as fin:
        rows = [line.strip().split (sep) for line in fin]
    
    df = pd.DataFrame (rows[1:], columns=rows[0])
    return df

def modify_context (paths, row):
    book_id = row['book_id']
    last_token = max (row['persons_end_token'], row['locations_end_token'])
    filename = correct_path (paths, book_id)
    df = read_file (filename, sep='\t')
    df = df.query ('token_ID_within_document == @last_token')
    sent_id = df['sentence_ID']
    return sent_id

def main (args):
    examples = pd.read_csv (args.sample_file, sep='\t')
    examples.head (20)
    examples['sent_id'] = examples.apply (lambda x: modify_context (args.paths, x), axis=1)
    print (examples.head (5))

if __name__ == "__main__":
    main (readArgs ())
