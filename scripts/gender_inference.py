""" Extract the inferential gender for the characters in each book."""

import argparse
import os
import glob
import json
import pandas as pd
from tqdm import tqdm

def readArgs ():
    parser = argparse.ArgumentParser (description="Collects the inferred gender for named entities")
    parser.add_argument ("--booknlp-dirs", required=True, type=str, nargs="+", help="Directories contain bookNLP data for individual titles")
    parser.add_argument ("--output-file", required=True, type=str, help="Flat file contains the inferred gender values")
    args = parser.parse_args ()
    return args


def read_inferred_genders_from_file (dirnames, cols=[]):
    """ Read the inferred genders from the file.

    Arguments:
    dirnames(list): A list of the directory names that contain individual book files.
    cols(list): A list of column names (str)
    """

    rows = list ()
    rows.append (cols)

    for dirname in dirnames:
        for filename in tqdm (glob.glob (os.path.join (dirname, f"*.book"))):
            book_id = ''.join(os.path.basename (filename).split (".")[0:-1])
            with open (filename) as fin:
                js = json.load (fin)

            for char in js["characters"]:
                if char["g"] is not None:
                    rows.append ([book_id, \
                                  char["id"], \
                                  char["g"]["argmax"], \
                                  char["g"]["max"]])
                    
    df = pd.DataFrame (rows[1:], columns=rows[0])
    return df

def main (args):
    df = read_inferred_genders_from_file (args.booknlp_dirs, cols=["book_id", "char_id", "inf_gender", "prob"])
    df.to_csv (args.output_file, sep="\t", header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())
