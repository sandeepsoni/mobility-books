import argparse
import torch
torch.manual_seed (96)

from spatial_relation_classes import TempRelationPrediction

def readArgs ():
	parser = argparse.ArgumentParser (description="Script to train and evaluate a spatial relation prediction model")
	parser.add_argument ("--pretrained-model-name", required=True, type=str, help="Name of the pretrained model")
	parser.add_argument ("--training-data-file", required=True, type=str, help="Training data is in this pickle file")
	parser.add_argument ("--num-epochs", required=False, default=10, type=int, help="Number of epochs for training")
	parser.add_argument ("--context-field", required=False, default="context_100", type=str, help="Column name that contains the entire text")
	parser.add_argument ("--depvar-field", required=False, default="Narrative Tense", type=str, help="Column name that maps to the dependent variable")
	parser.add_argument ("--model-path", required=True, type=str, help="Path to the file that will store the model")
	parser.add_argument ("--num-labels", required=False, type=int, default=2, help="The number of labels in the spatial relation prediction task")
	args = parser.parse_args ()
	return args

def main (args):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	srp  = TempRelationPrediction (model_name=args.pretrained_model_name,
									  bert_dims=768, 
									  n_labels=args.num_labels,
									  device=device,
									  lr=1e-6)
	srp.load_training_data (args.training_data_file, filter_field=args.depvar_field)
	srp.start_training (num_epochs=args.num_epochs, context_field=args.context_field, depvar_field=args.depvar_field, verbose=True)
	srp.save_model (args.model_path)

if __name__ == "__main__":
	main (readArgs ())
