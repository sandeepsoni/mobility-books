import argparse
from spatial_relation_classes import SpatialRelationPrediction
import torch
import pandas as pd
import os

def readArgs ():
	parser = argparse.ArgumentParser (description="Apply the trained model on a bunch of examples from unseen books")
	parser.add_argument ("--binary-model-path", type=str, required=True, help="Path to the binary classifier")
	parser.add_argument ("--spatial-model-path", type=str, required=True, help="Path to the spatial classifier")
	parser.add_argument ("--book-id", type=str, required=True, help="Gutenberg ID of the book")
	parser.add_argument ("--collocations-dir", type=str, required=True, help="Path to the corpus containing collocations")
	parser.add_argument ("--output-dir", type=str, required=True, help="Path to the output file")
	args = parser.parse_args ()
	return args

def main (args):
	spatial_labels = [ "NO RELATIONSHIP ASSERTED",
					   "TOWARD(got there)",
					   "FROM",
					   "NEAR",
					   "IN",
					   "NEGATIVE ASSERTION",
					   "THROUGH",
					   "TOWARD (uncertain got there)",
					   "UNCERTAIN ASSERTION"]
        
	accepted_labels = ["GOOD", "BAD"]
	book_df = pd.read_csv (os.path.join (args.collocations_dir, f"{args.book_id}.collocations"), sep="\t")
	# rename columns
	book_df.rename(columns = {'persons_start_token':'persons_start', 'persons_end_token':'persons_end','locations_start_token':'locations_start', 'locations_end_token': 'locations_end'}, inplace = True)
	checkpoint = torch.load (args.binary_model_path)
	binary_model = SpatialRelationPrediction(n_labels=2)
	binary_model.load_state_dict(checkpoint["model_state_dict"])
	predictions = binary_model.evaluate_book (book_df)
	predictions = [accepted_labels[prediction] for prediction in predictions]
	book_df["binary_classifier_predictions"] = predictions
	book_df = book_df[book_df["binary_classifier_predictions"] == "GOOD"]

	# load the spatial model
	checkpoint = torch.load (args.spatial_model_path)
	spatial_model = SpatialRelationPrediction (n_labels=9)
	spatial_model.load_state_dict (checkpoint["model_state_dict"])
	predictions = spatial_model.evaluate_book (book_df)
	predictions = [spatial_labels[prediction] for prediction in predictions]
	book_df["spatial_classifier_predictions"] = predictions
	os.makedirs (args.output_dir, exist_ok=True)
	book_df.to_csv (os.path.join (args.output_dir, f"{args.book_id}.predictions"), sep="\t", header=True, index=False)

if __name__ == "__main__":
	main (readArgs ())


