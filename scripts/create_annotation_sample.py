"""

Usage:
Author: Sandeep Soni
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
import sys

from ..modules.samplers import ReservoirSampler

def gen_random_string (length=6):
    return ''.join ([choice (ascii_uppercase) for i in range (length)])

def readArgs ():
    parser = argparse.ArgumentParser (description="Creates an annotation sample for name x location")
    parser.add_argument ("--collocation-dir", required=True, type=str, help="Directory contains collocation files")
    parser.add_argument ("--content-dir", required=True, type=str, nargs="+", help="Directory contains the content files")
    parser.add_argument ("--sample-size", required=False, type=int, default=10000, help="Size of one sample")
    parser.add_argument ("--batch-size", required=False, type=int, default=1000, help="Size of one batch")
    parser.add_argument ("--context-window-sizes", required=False, type=int, nargs="+", default=[10], help="The size of the context to show during annotation")
    parser.add_argument ("--output-file", required=True, type=str, help="The output file contains the sample")
    args = parser.parse_args ()
    return args

def gen_random_examples (collocation_dir, num_samples=10000, sep="\t"):
    rs = ReservoirSampler (num_samples)
    for filename in tqdm (glob.glob (os.path.join (collocation_dir, "*.collocations"))):
        with open (filename) as fin:
            for i, line in enumerate (fin):
                if i == 0:
                    header = line.strip().split (sep)
                if i > 0:
                    rs.add (line)

    ids = [gen_random_string () for item in rs.reservoir]

    return rs, ids, header

def main (args):
    rs, ids, header = gen_random_examples (args.collocation_dir, num_samples=args.sample_size)

    rows = list ()
    for i, item in enumerate (rs.reservoir):
        parts = item.strip().split ("\t")
        parts.append (ids[i])
        rows.append (parts)

    output = list ()
    header.append ("ID")
    df = pd.DataFrame (rows, columns=header)
    for i,row in tqdm (df.iterrows ()):
        book_id = row["book_id"]
        start = min(int (row["persons_start_token"]), int (row["locations_start_token"]))
        end = max(int (row["persons_end_token"]), int(row["locations_end_token"])) 
        special_token_positions = {int (row["persons_start_token"]), 
                                   int (row["persons_end_token"]),
                                   int (row["locations_start_token"]),
                                   int (row["locations_end_token"])}

        for context_window in args.context_window_sizes:
            from_here = start - context_window
            till_there = end + context_window

            for content_dir in args.content_dir:
                path = os.path.join (content_dir, f"{book_id}.tokens")
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break

            target_sent_id = sys.maxsize
            with open (path) as fin:
                context = list ()
                for j, line in enumerate (fin):
                    if j == 0:
                        continue
                    parts = line.strip().split ("\t")
                    sent_id = int (parts[1]) # get the sentence ID
                    token_id = int (parts[3]) # get the token ID
                    if token_id < from_here:
                        continue
                    elif token_id >= from_here and token_id <= till_there:
                        if token_id not in special_token_positions:
                            if sent_id <= (target_sent_id+1): # Consider the sentence after the sentence that contained the last mention
                                context.append (parts[4]) # append the token
                            else:
                                break
                        else:
                            if token_id == int (row["persons_start_token"]):
                                context.append ("<char>")
                            if token_id == int (row["locations_start_token"]):
                                context.append ("<place>")

                            context.append (parts[4])
                            
                            if token_id == int (row["persons_end_token"]):
                                context.append ("</char>")
                            if token_id == int (row["locations_end_token"]):
                                context.append ("</place>")

                            if token_id == end:
                                target_sent_id = int (parts[1])
                    else:
                        break

                context = " ".join (context)
            row[f"context_{context_window}"] = context
        output.append (row)

    output_df = pd.DataFrame (output)
    # rename some columns
    output_df = output_df.rename (columns={"persons_start_token": "char_start_token", 
                                           "persons_end_token": "char_end_token", 
                                           "persons_text": "char_text",
                                           "persons_coref": "char_coref",
                                           "persons_tag": "char_tag",
                                           "persons_cat": "char_cat",
                                           "locations_start_token": "place_start_token",
                                           "locations_end_token": "place_end_token",
                                           "locations_text": "place_text",
                                           "locations_coref": "place_coref",
                                           "locations_tag": "place_tag",
                                           "locations_cat": "place_cat"})

    output_df["valid_relation"] = ""
    output_df["spatial_relation"] = ""
    output_df["temporal_span"] = ""
    output_df["narrative_tense"] = ""
    output_df["comments"] = ""
    output_df.to_csv (args.output_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
