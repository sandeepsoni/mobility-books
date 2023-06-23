import pandas as pd
import argparse
import os, glob
from tqdm import tqdm
import json

def readArgs ():
    parser = argparse.ArgumentParser (description="Extract all toponym candidates and then do resolution")
    parser.add_argument ("--booknlp-dir", type=str, required=True, help="Directory contains the booknlp files")
    parser.add_argument ("--cat", type=str, required=False, default="GPE", help="NE categories")
    parser.add_argument ("--gazetteer-file", type=str, required=True, help="File contains the gazeteer")
    parser.add_argument ("--output-json-file", type=str, required=True, help="File contains the toponym candidates and their resolutions")
    args = parser.parse_args ()
    return args

def get_entities_from_book (filename):
    df = pd.read_csv (filename, sep="\t", on_bad_lines='skip', engine="python")
    return df

def main (args):
    # Read all the locations from the entities file
    """
    all_locs = list ()
    for filename in tqdm (glob.glob (os.path.join(args.booknlp_dir, "*.entities"))):
        entities_df = get_entities_from_book (filename)
        book_id = os.path.basename (filename)[:-len(".entities")]
        entities_df["book_id"] = book_id
        locs_df = entities_df.query ('cat == @args.cat')
        all_locs.append (locs_df)

    all_locs = pd.concat (all_locs)
    print (len (all_locs))
    """
    # Read all entries from the gazeteer
    gaz = pd.read_csv (args.gazetteer_file, sep='\t', quoting=True, quotechar='"')
    print (len (gaz))
    # Output a JSON files

if __name__ == "__main__":
    main (readArgs ())