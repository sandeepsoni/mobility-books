import argparse
from spatial_relation_classes import SpatialRelationPrediction
import torch

def readArgs ():
	parser = argparse.ArgumentParser (description="Apply the trained model on a bunch of examples from unseen books")
	parser.add_argument ("--binary-model-path", type=str, required=True, help="Path to the binary classifier")
	parser.add_argument ("--spatial-model-path", type=str, required=True, help="Path to the spatial classifier")
	parser.add_argument ("--book-id", type=str, required=True, help="Gutenberg ID of the book")
	parser.add_argument ("--corpus-dir", type=str, required=True, help="Path to the corpus containing the Gutenberg corpus")
	parser.add_argument ("--output-path", type=str, required=True, help="Path to the output file")
	args = parser.parse_args ()
	return args

def main (args):
	checkpoint = torch.load (args.binary_model_path)
	model = SpatialRelationPrediction(n_labels=2)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

if __name__ == "__main__":
	main (readArgs ())


