import argparse
from spatial_relation_classes import SpatialRelationPrediction
import torch
import os
import pandas as pd
from tqdm import tqdm

def readArgs ():
    parser = argparse.ArgumentParser (description="Apply the trained model on a bunch of examples from unseen books")
    parser.add_argument ("--book-id", type=str, required=True, help="Gutenberg ID of the book")
    parser.add_argument ("--collocations-dir", type=str, required=True, help="Directory that contains collocations")
    parser.add_argument ("--content-dir", type=str, required=True, nargs="+", help="Path to the corpus containing the Gutenberg corpus")
    parser.add_argument ("--context-window-size", type=int, required=False, default=100, help="The size of the context window")
    parser.add_argument ("--output-dir", type=str, required=True, help="Directory contains output file")
    args = parser.parse_args ()
    return args

def main (args):
    #checkpoint = torch.load (args.binary_model_path)
    #model = SpatialRelationPrediction(n_labels=2)
    #model.load_state_dict(checkpoint["model_state_dict"])
    #model.eval()

    df = pd.read_csv (os.path.join (args.collocations_dir, f"{args.book_id}.collocations"), sep="\t")
    output = list ()
    for i,row in tqdm (df.iterrows ()):
        book_id = row["book_id"]
        start = min(int (row["persons_start_token"]), int (row["locations_start_token"]))
        end = max(int (row["persons_end_token"]), int(row["locations_end_token"])) 

        context_window = args.context_window_size
        from_here = start - context_window
        till_there = end + context_window

        book_found = False
        for content_dir in args.content_dir:
            path = os.path.join (content_dir, f"{book_id}.tokens")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                book_found = True
                break

        if not book_found: continue

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
    output_df.to_csv (os.path.join (args.output_dir, f"{args.book_id}.collocations"), sep="\t", index=False, header=True)

if __name__ == "__main__":
	main (readArgs ())
