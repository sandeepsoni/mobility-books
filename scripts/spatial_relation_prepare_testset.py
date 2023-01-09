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


def search_sublist (sentence_as_tokens, entity_as_tokens):
	"""
	For the entity span (decomposed into tokens), find all 
	the start and end positions within the 
	sentence (decomposed into tokens)
	"""
	sublist_positions = list ()
	for i, token in enumerate (sentence_as_tokens):
		if token == entity_as_tokens[0]: # see if the first character matches
			if all ([tok == sentence_as_tokens[i+j] if (i+j) < len(sentence_as_tokens) else False for j,tok in enumerate (entity_as_tokens)]):
				sublist_positions.append ((i, i+len(entity_as_tokens)))

	return sublist_positions

def get_start_end_indices (toks, ent_toks, window_size=100):
	""" Helper function that gives the start and end indices of the `ent_toks` in `toks`

	toks (list): All the tokens as a list.
	ent_toks (list): All the tokens within the entities.
	window_size (int): The size of the window.

	Returns the index positions of the start and the end tokens for the entity.
	"""
	found = False
	if toks[window_size] == ent_toks[0]:
		found = True
		for i in range (1, len (ent_toks)):
			if not toks[window_size+i] == ent_toks[i]:
				found = False
				break
	if found:
		start = window_size
		end = start + len (ent_toks) - 1
		return start, end
	elif toks[-(window_size+1)] == ent_toks[-(1)]:
		found = True
		for i in range (1, len (ent_toks)):
			if not toks[-(window_size+1+i)] == ent_toks[-(i+1)]:
				found = False
				break   
	if found:
		end = len (toks) - (window_size + 1)
		start = end - len (ent_toks) + 1
		return start, end
	else:
		print ("Something is wrong")
		return -1, -1

def mark_start_and_end (frame, window_size):
	""" mark the start and the end indices of the entities in each example of the annotations.

	annotations (pd.DataFrame): All the annotations.
	window_size (int): The context size.

	Returns pandas Dataframe that contains additional columns with start and end indices.
	"""

	rows = list ()
	for i in range (len (frame)):
		toks = frame.iloc[i][f"context_{window_size}"].split(" ")
		per_text = frame.iloc[i]["persons_text"]
		loc_text = frame.iloc[i]["locations_text"]
		loc_toks = loc_text.split (" ")
		per_toks = per_text.split (" ")

		# Check if per_text is a substring of loc_text
		if len (search_sublist (loc_toks, per_toks)) > 0:
			loc_start, loc_end = get_start_end_indices (toks, loc_toks, window_size=window_size)
			positions = search_sublist (loc_toks, per_toks)
			per_start = loc_start + positions[0][0]
			per_end = loc_start + positions[0][1]
			frame.iloc[i]["persons_start"] = per_start
			frame.iloc[i]["persons_end"] = per_end
			frame.iloc[i]["locations_start"] = loc_start
			frame.iloc[i]["locations_end"] = loc_end
		# Check if loc_text is a substring of per_text    
		elif len (search_sublist (per_toks, loc_toks)) > 0:
			per_start, per_end = get_start_end_indices (toks, per_toks, window_size=window_size)
			positions = search_sublist (per_toks, loc_toks)
			loc_start = per_start + positions[0][0]
			loc_end = loc_start + positions[0][1]
			frame.iloc[i]["persons_start"] = per_start
			frame.iloc[i]["persons_end"] = per_end
			frame.iloc[i]["locations_start"] = loc_start
			frame.iloc[i]["locations_end"] = loc_end
		else:
			per_start, per_end = get_start_end_indices (toks, per_text.split(" "), window_size=window_size)
			loc_start, loc_end = get_start_end_indices (toks, loc_text.split(" "), window_size=window_size)
			if not " ".join(toks[per_start:per_end+1]) == per_text or not " ".join(toks[loc_start:loc_end+1]) == loc_text:
				continue
			else:
				frame.iloc[i]["persons_start"] = per_start
				frame.iloc[i]["persons_end"] = per_end
				frame.iloc[i]["locations_start"] = loc_start
				frame.iloc[i]["locations_end"] = loc_end
        
		rows.append ([per_text, 
					  loc_text, 
					  frame.iloc[i][f"context_{window_size}"],
					  per_start, 
					  per_end, 
					  loc_start, 
					  loc_end])
    
	return pd.DataFrame (rows, 
						 columns=["persons_text",
                                 "locations_text",
                                 f"context_{window_size}",
                                 "persons_start",
                                 "persons_end",
                                 "locations_start",
                                 "locations_end"])


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
	output_df = mark_start_and_end (output_df, window_size=args.context_window_size)
	output_df.to_csv (os.path.join (args.output_dir, f"{args.book_id}.examples"), sep="\t", index=False, header=True)

if __name__ == "__main__":
	main (readArgs ())
