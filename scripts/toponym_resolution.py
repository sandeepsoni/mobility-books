import pandas as pd
import argparse
import os, glob
from tqdm import tqdm
import json
import unidecode
import re

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

def format_into_query (toponym_candidate):
    # remove accents
    toponym_candidate = unidecode.unidecode(toponym_candidate) #remove accents
    toponym_candidate = toponym_candidate.lower() # lowercase
    toponym_candidate = re.sub('[^0-9a-zA-Z]+', ' ', toponym_candidate) # replace non-alphanumerics by space
    toponym_candidate = ' '.join(toponym_candidate.split()) # replace multiple whitespaces with a single whitespace
    return toponym_candidate

def create_json (book_id, coref_id, start_token, end_token, text, query, num_matches):
    js = {}
    js["book_id"] = book_id
    js["coref_id"] = coref_id
    js["start_token"]= start_token
    js["end_token"] = end_token
    js["text"] = text
    js["query"] = query
    js["num_matches"] = num_matches
    return js


def main (args):
    # Read all the locations from the entities file
    all_locs = list ()
    for filename in tqdm (glob.glob (os.path.join(args.booknlp_dir, "*.entities"))):
        entities_df = get_entities_from_book (filename)
        book_id = os.path.basename (filename)[:-len(".entities")]
        entities_df["book_id"] = book_id
        locs_df = entities_df.query ('cat == @args.cat')
        all_locs.append (locs_df)
        break

    all_locs = pd.concat (all_locs)
    all_locs["query_text"] = all_locs["text"].apply (lambda x: format_into_query (x))

    # Read all entries from the gazeteer
    gaz = pd.read_csv (args.gazetteer_file, sep='\t', quoting=True, quotechar='"')
    gaz = gaz.query ('lang == "en"')
    gaz = gaz[["text_string", "place_id", "lat", "lon"]]
    print (len (gaz))
    # Output a JSON files
    with open (args.output_json_file, "w") as fout:
        for i, loc in all_locs.iterrows ():
            q = loc["query_text"]
            df = gaz.query ('text_string == @q')
            num_matches = len (df)
            js = create_json (loc["book_id"], loc["COREF"], loc["start_token"], loc["end_token"], loc["text"], q, num_matches)
            fout.write (f"{json.dumps (js)}\n")


if __name__ == "__main__":
    main (readArgs ())