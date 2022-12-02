""" Script to identify small, highly mobile books that are narrated in the third person.
"""
import os, glob
import argparse
import json
import pandas as pd
import numpy as np

def readArgs ():
    parser = argparse.ArgumentParser (description="Identify highly mobile books written in third person")
    parser.add_argument ("--input-filename", type=str, required=True, help="File contains the tokens in the book")
    parser.add_argument ("--output-dir", type=str, required=True, help="Directory contains the output file")
    args = parser.parse_args ()
    return args

def get_rows_of_book (filename):
    df = pd.read_csv (filename, sep="\t", on_bad_lines='skip', engine="python")
    return df

def main (args):
    os.makedirs (args.output_dir, exist_ok=True)
    book_id = os.path.basename (args.input_filename).split (".")[0]
    output_filename = os.path.join (args.output_dir, f"{book_id}.third_person_heuristics.json")
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        return
    entities = get_rows_of_book (os.path.join (os.path.dirname (args.input_filename), f"{book_id}.entities"))
    locations = entities[entities["cat"].isin (["GPE"])]
    tokens = get_rows_of_book (args.input_filename)
    first_person_tokens = tokens[tokens["word"].isin (["I"])]
    quotes = get_rows_of_book (os.path.join (os.path.dirname (args.input_filename), f"{book_id}.quotes"))
    
    # Iterate over each token position, and count only if it is outside of all quote spans.
    token_positions = first_person_tokens["token_ID_within_document"].values
    quote_starts = quotes["quote_start"].values
    quote_ends = quotes["quote_end"].values
    count_non_quoted_i = 0
    for token_position in token_positions:
        if not np.any(np.logical_and (token_position > quote_starts, token_position < quote_ends)):
            count_non_quoted_i += 1

    js = {}
    js["book_id"] = book_id
    js["prob_gpe"] = len (locations) / (len (entities) + 1e-5)
    js["n_tokens"] = len (tokens)
    js["count_non_quoted_i"] = count_non_quoted_i

    with open (output_filename, "w") as fout:
        fout.write (json.dumps (js))

if __name__ == "__main__":
    main (readArgs ())
