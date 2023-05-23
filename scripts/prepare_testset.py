import argparse
import pandas as pd
import os, sys
from tqdm import tqdm

def readArgs ():
    parser = argparse.ArgumentParser (description="Apply the trained model on a bunch of examples from unseen books")
    parser.add_argument ("--book-id", type=str, required=True, help="ID of the book")
    parser.add_argument ("--collocations-dir", type=str, required=True, help="Directory that contains collocations")
    parser.add_argument ("--content-dir", type=str, required=True, nargs="+", help="Path to the corpus containing the content of each book")
    parser.add_argument ("--context-window-size", type=int, required=False, default=100, help="The size of the context window")
    parser.add_argument ("--output-dir", type=str, required=True, help="Directory contains output file")
    args = parser.parse_args ()
    return args

def main (args):
    os.makedirs (args.output_dir, exist_ok=True)
    df = pd.read_csv (os.path.join (args.collocations_dir, f"{args.book_id}.collocations"), sep="\t")
    output = list ()
    for i,row in tqdm (df.iterrows ()):
        book_id = row["book_id"]
        start = min(int (row["persons_start_token"]), int (row["locations_start_token"]))
        end = max(int (row["persons_end_token"]), int(row["locations_end_token"])) 
        special_token_positions = {int (row["persons_start_token"]), 
                                   int (row["persons_end_token"]),
                                   int (row["locations_start_token"]),
                                   int (row["locations_end_token"])}
        
        from_here = start - args.context_window_size
        till_there = end + args.context_window_size

        # Search for file in multiple directories
        book_found = False
        for content_dir in args.content_dir:
            path = os.path.join (content_dir, f"{book_id}.tokens")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                book_found = True
                break
        
        if not book_found: continue

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
        row[f"context_{args.context_window_size}"] = context
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
    
    # add an ID column
    output_df.loc[:, "ID"] = output_df.index
    
    output_df.to_csv (os.path.join (args.output_dir, f"{args.book_id}.collocations"), sep="\t", index=False, header=True)
    output_df = output_df[['book_id', "ID", f'context_{args.context_window_size}']]
    output_df.to_csv (os.path.join (args.output_dir, f"{args.book_id}.examples"), sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())