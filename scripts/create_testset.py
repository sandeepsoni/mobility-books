""" Extract examples for a set of Project Gutenberg books.
"""
import argparse
import os
import glob
import pandas as pd
from random import choice
from string import ascii_uppercase
from functools import reduce
from tqdm import tqdm
import random
import logging

def readArgs ():
    parser = argparse.ArgumentParser (description="Creates an annotation sample for name x location")
    parser.add_argument ("--collocation-dir", required=True, type=str, help="Directory contains collocation files")
    parser.add_argument ("--content-dir", required=True, type=str, nargs="+", help="Directory contains the content files")
    parser.add_argument ("--testset-bookids-file", required=True, type=str, help="File contains the IDs of the books to be included in the test set")
    parser.add_argument ("--context-window-size", required=False, type=int, default=100, help="The size of the context to show during annotation")
    parser.add_argument ("--output-file", required=True, type=str, help="The output file contains the sample")
    args = parser.parse_args ()
    return args

def main (args):
    collocations = list ()
    for filename in glob.glob(os.path.join (args.collocation_dir, "*.collocations")):
        collocation_book = pd.read_csv (filename, sep="\t")
        collocations.append (collocation_book)
    df = pd.concat (collocations)
    logging.info (len (df))

    book_ids = set ()
    with open (args.testset_bookids_file) as fin:
        for line in fin:
            book_ids.add (int (line.strip()))

    output = list ()
    context_window = args.context_window_size
    for i,row in tqdm (df.iterrows ()):
        book_id = row["book_id"]
        if book_id not in book_ids:
            continue
        start = min(int (row["persons_start_token"]), int (row["locations_start_token"]))
        end = max(int (row["persons_end_token"]), int(row["locations_end_token"])) 

        from_here = start - context_window
        till_there = end + context_window

        # Find which directory the book content is present
        for content_dir in args.content_dir:
            path = os.path.join (content_dir, f"{book_id}.tokens")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break

        # Now open that file and get the overall context.
        with open (path) as fin:
            context = list ()
            for j, line in enumerate (fin):
                if j == 0:
                    continue
                parts = line.strip().split ("\t")
                token_id = int (parts[3])
                if token_id < from_here:
                    continue
                elif token_id >= from_here and token_id <= till_there:
                    context.append (parts[4])
                else:
                    break

            context = " ".join (context)
        row[f"context_{context_window}"] = context
        output.append (row)

    output_df = pd.DataFrame (output)
    output_df.to_csv (args.output_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
