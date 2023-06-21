import os, glob
import pandas as pd
from tqdm import tqdm

import argparse

def readArgs ():
    parser = argparse.ArgumentParser (description="Find cooccurrences of names and locations")
    parser.add_argument ("--input-filename", type=str, required=True, help="File contains all the entities")
    parser.add_argument ("--output-dir", type=str, required=True, help="Directory contains the output files")
    parser.add_argument ("--window-size", type=int, required=False, default=10, help="size of the window for cooccurrences")
    args = parser.parse_args ()
    return args

def get_entities_from_book (filename):
    df = pd.read_csv (filename, sep="\t", on_bad_lines='skip', engine="python")
    return df

def get_colocations (persons_df, locations_df, within_window=10):
    df = pd.concat ([persons_df, locations_df])
    frame = df.sort_values (by="start_token")
    frame = frame.reset_index()
    rows = list ()
    for index, row in frame.iterrows ():
        if row["cat"] == "PER":
            s = row["start_token"]
            e = row["end_token"]
            i = index
            while i > 0:
                if frame.iloc[i]["cat"] in ["GPE", "FAC", "LOC"]:
                    if abs(frame.iloc[i]["start_token"] - e) <= within_window:
                        rows.append ([s, 
                                      e,
                                      row["text"], 
                                      row["COREF"], 
                                      row["cat"], 
                                      row["prop"],
                                      frame.iloc[i]["start_token"], 
                                      frame.iloc[i]["end_token"], 
                                      frame.iloc[i]["text"], 
                                      frame.iloc[i]["COREF"],
                                      frame.iloc[i]["cat"],
                                      frame.iloc[i]["prop"]
                                      ])
                    else:
                        break
                i -=1
            i = index + 1
            while i < len (df):
                if frame.iloc[i]["cat"] in ["GPE", "FAC", "LOC"]:
                    if abs(frame.iloc[i]["start_token"] - e) <= within_window:
                        rows.append ([s, 
                                      e,
                                      row["text"], 
                                      row["COREF"], 
                                      row["cat"], 
                                      row["prop"],
                                      frame.iloc[i]["start_token"], 
                                      frame.iloc[i]["end_token"], 
                                      frame.iloc[i]["text"], 
                                      frame.iloc[i]["COREF"],
                                      frame.iloc[i]["cat"],
                                      frame.iloc[i]["prop"]
                                      ])
                    else:
                        break
                i+=1
                
    return pd.DataFrame (rows, columns=["persons_start_token",
                                        "persons_end_token",
                                        "persons_text",
                                        "persons_coref",
                                        "persons_cat",
                                        "persons_tag",
                                        "locations_start_token",
                                        "locations_end_token",
                                        "locations_text",
                                        "locations_coref",
                                        "locations_cat",
                                        "locations_tag"])

def main (args):
    os.makedirs (args.output_dir, exist_ok=True)
    book_id = ".".join(os.path.basename (args.input_filename).split (".")[0:-1])
    output_filename = os.path.join (args.output_dir, f"{book_id}.collocations")
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        return
    df = get_entities_from_book (args.input_filename)
    persons = df[df["cat"] == "PER"]
    locations = df[df["cat"].isin (["GPE", "FAC", "LOC"])]
    collocations = get_colocations (persons, locations, within_window=args.window_size)
    collocations["book_id"] = book_id
    collocations.to_csv (output_filename, sep="\t", header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())
