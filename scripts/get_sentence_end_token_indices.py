"""
Amanpreet's annotations did not have the sentence ends. 
So for all the examples in the sample that Amanpreet used, we'll try to get the sentence end token indices.

python get_sentence_end_token_indices.py \
    --dir-paths /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_1_4/ /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_5_9/ \
    --sample-file ../data/annotations/sample_annotations.v1.tsv \
    --output-file ../data/annotations/sample_annotations.v1.modified.tsv
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

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
    df = df.astype({
        'paragraph_ID':'int64',
        'sentence_ID': 'int64',
        'token_ID_within_sentence': 'int64',
        'token_ID_within_document': 'int64'
    })
    return df

def modify_context (paths, row, offset=10):
    book_id = row['book_id']
    last_token = max (row['persons_end_token'], row['locations_end_token'])
    filename = correct_path (paths, f"{book_id}.tokens")
    df = read_file (filename, sep='\t')
    next_sent = df.query ('token_ID_within_document == @last_token')['sentence_ID'].values[0] + 1
    token_next_sent_end =  df.query ('sentence_ID == @next_sent')['token_ID_within_document'].max()
    field_name = f'context_{offset}'
    end = min (offset + last_token, token_next_sent_end)
    start = min (row['persons_start_token'], row['locations_end_token']) - offset
    text = row[field_name]
    tokens = text.split ()
    return " ".join (tokens[0:end-start+1])

def main (args):
    tqdm.pandas(desc='My bar!')
    examples = pd.read_csv (args.sample_file, sep='\t')
    examples['context_10'] = examples.progress_apply (lambda x: modify_context (args.dir_paths, x, offset=10), axis=1)
    examples['context_50'] = examples.progress_apply (lambda x: modify_context (args.dir_paths, x, offset=50), axis=1)
    examples['context_100'] = examples.progress_apply (lambda x: modify_context (args.dir_paths, x, offset=100), axis=1)

    examples.to_csv (args.output_file, sep='\t', index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
