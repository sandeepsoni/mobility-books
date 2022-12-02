import argparse
import os
import glob
import pandas as pd
from random import choice
from string import ascii_uppercase
from functools import reduce
from tqdm import tqdm
import random

class ReservoirSampler ():
    def __init__ (self, k):
        self.reservoir = list ()
        self.max_limit = k
        self.currently_filled = 0

    def __len__ (self):
        return len (self.reservoir)

    def add (self, new_item):
        if self.currently_filled < self.max_limit:
            # case when the reservoir is not filled completely.
            # in this case, we simply copy the element to the reservoir
            self.reservoir.append (new_item)
        else:
            # case when the reservoir is filled completely.
            # in this case, we decide if we want to ignore the new item or
            # replace an existing item with the new item
            j = random.randrange(self.currently_filled)
            # if the randomly picked index is smaller than the reservoir size
            # then replace the item present at the index with the new item;
            # otherwise, ignore the new item.
            if j < self.max_limit:
                self.reservoir[j] = new_item

        self.currently_filled += 1

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

def main (args):
    rs = ReservoirSampler (args.sample_size)
    for filename in tqdm (glob.glob (os.path.join (args.collocation_dir, "*.collocations"))):
        with open (filename) as fin:
            for i, line in enumerate (fin):
                if i == 0:
                    header = line.strip().split ("\t")
                if i > 0:
                    rs.add (line)

    ids = [gen_random_string () for item in rs.reservoir]

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

        for context_window in args.context_window_sizes:
            from_here = start - context_window
            till_there = end + context_window

            for content_dir in args.content_dir:
                path = os.path.join (content_dir, f"{book_id}.tokens")
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break

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
    output_df["is_character_correct"] = ""
    output_df["is_location_correct"] = ""
    output_df["is_character_at_location"] = ""
    output_df.to_csv (args.output_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
