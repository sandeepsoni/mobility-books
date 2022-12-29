import pandas as pd
import pickle
import argparse
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_annotations_from_file (filename, sep="\t"):
	rows = list ()
	with open (filename) as fin:
		for line in fin:
			parts = line.strip().split (sep)
			rows.append (parts)
	df = pd.DataFrame (rows[1:], columns=rows[0])
	return df

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

def mark_start_and_end (annotations, window_size):
	""" mark the start and the end indices of the entities in each example of the annotations.

	annotations (pd.DataFrame): All the annotations.
	window_size (int): The context size.

	Returns pandas Dataframe that contains additional columns with start and end indices.
	"""

	rows = list ()
	for i in range (len (annotations)):
		toks = annotations.iloc[i][f"context_{window_size}"].split(" ")
		per_text = annotations.iloc[i]["persons_text"]
		loc_text = annotations.iloc[i]["locations_text"]
		loc_toks = loc_text.split (" ")
		per_toks = per_text.split (" ")

		# Check if per_text is a substring of loc_text
		if len (search_sublist (loc_toks, per_toks)) > 0:
			loc_start, loc_end = get_start_end_indices (toks, loc_toks, window_size=window_size)
			positions = search_sublist (loc_toks, per_toks)
			per_start = loc_start + positions[0][0]
			per_end = loc_start + positions[0][1]
			annotations.iloc[i]["persons_start"] = per_start
			annotations.iloc[i]["persons_end"] = per_end
			annotations.iloc[i]["locations_start"] = loc_start
			annotations.iloc[i]["locations_end"] = loc_end
		# Check if loc_text is a substring of per_text    
		elif len (search_sublist (per_toks, loc_toks)) > 0:
			per_start, per_end = get_start_end_indices (toks, per_toks, window_size=window_size)
			positions = search_sublist (per_toks, loc_toks)
			loc_start = per_start + positions[0][0]
			loc_end = loc_start + positions[0][1]
			annotations.iloc[i]["persons_start"] = per_start
			annotations.iloc[i]["persons_end"] = per_end
			annotations.iloc[i]["locations_start"] = loc_start
			annotations.iloc[i]["locations_end"] = loc_end
		else:
			per_start, per_end = get_start_end_indices (toks, per_text.split(" "), window_size=window_size)
			loc_start, loc_end = get_start_end_indices (toks, loc_text.split(" "), window_size=window_size)
			if not " ".join(toks[per_start:per_end+1]) == per_text or not " ".join(toks[loc_start:loc_end+1]) == loc_text:
				continue
			else:
				annotations.iloc[i]["persons_start"] = per_start
				annotations.iloc[i]["persons_end"] = per_end
				annotations.iloc[i]["locations_start"] = loc_start
				annotations.iloc[i]["locations_end"] = loc_end
        
		rows.append ([annotations.iloc[i]["ID"], 
					  per_text, 
					  loc_text, 
					  annotations.iloc[i][f"context_{window_size}"],
					  annotations.iloc[i]["Spatial Relation"],
					  annotations.iloc[i]["Temporal Span"],
					  annotations.iloc[i]["Narrative Tense"],
					  per_start, 
					  per_end, 
					  loc_start, 
					  loc_end])
    
	return pd.DataFrame (rows, 
						 columns=["ID", 
                                 "persons_text",
                                 "locations_text",
                                 f"context_{window_size}",
                                 "Spatial Relation",
                                 "Temporal Span",
                                 "Narrative Tense",
                                 "persons_start",
                                 "persons_end",
                                 "locations_start",
                                 "locations_end"])

def readArgs ():
	parser = argparse.ArgumentParser (description="Prepare training data for the model")
	parser.add_argument ("--annotations-file", type=str, required=True, help="The annotations file contains all the annotations")
	parser.add_argument ("--pickle-file", type=str, required=True, help="The pickle file contains all the annotations for fast loading")
	args = parser.parse_args ()
	return args

def main (args):
	annotations = read_annotations_from_file (args.annotations_file, sep="\t")
	logging.info (f"No. of annotations read from file: {len (annotations)}")

	new_annotations = dict ()
	for window_size in [10, 50, 100]:
		new_annotations[window_size] = mark_start_and_end (annotations, window_size=window_size)
		logging.info (f"No. of annotations which could be sanitized: {len (new_annotations[window_size])}")

	with open (args.pickle_file, "wb") as fout:
		pickle.dump (new_annotations, fout)

if __name__ == "__main__":
	main (readArgs())
