"""
Wrapper script to train a model for spatial relation prediction. 
"""
import argparse
import torch
torch.manual_seed (96)

from spatial_relation_classes import SpatialRelationPrediction
from sklearn.model_selection import train_test_split
from ..modules.relation_prediction_constants import ALL_LABELS, SPATIAL_LABELS, BAD_LABELS, VALID_LABELS

def preprocess_valid_relation_prediction (annotations, *args, **kwargs):
    labels = kwargs.get ("labels", ALL_LABELS)
    full_df = annotations[kwargs.get ("window_size", 10)]
    full_df = full_df.query ("`Spatial Relation` != ''")
    full_df = full_df.query ("`Spatial Relation` in @labels")
    full_df.loc[:,"Valid Relation"] = full_df["Spatial Relation"].apply (lambda x: VALID_LABELS[int(x in SPATIAL_LABELS)], axis=1)

    test_ids_file = kwargs.get ("test_ids_file", "")

    if test_ids_file == "":
        train_df, test_df = train_test_split(full_df,
                                             test_size=1-kwargs.get ("training_frac", .8),
                                             random_state=96)
    else:
        with open (test_ids_file) as fin:
            test_ids = [line for line in fin]
        train_df = full_df.query ("ID not in @test_ids")
        test_df = full_df.query ("ID in @test_ids")

    return full_df, train_df, test_df

def readArgs ():
	parser = argparse.ArgumentParser (description="Script to train and evaluate a spatial relation prediction model")
	parser.add_argument ("--pretrained-model-name", required=True, type=str, help="Name of the pretrained model")
	parser.add_argument ("--training-data-file", required=True, type=str, help="Training data is in this pickle file")
	parser.add_argument ("--test-ids-file", required=False, default="", type=str, required="Test file contains IDS on which we want to test")
	parser.add_argument ("--training-frac", required=False, default=0.8, type=float, help="Training fraction")
	parser.add_argument ("--num-epochs", required=False, default=10, type=int, help="Number of epochs for training")
	parser.add_argument ("--context-field", required=False, default="context_100", type=str, help="Column name that contains the entire text")
	parser.add_argument ("--model-path", required=True, type=str, help="Path to the file that will store the model")
	parser.add_argument ("--num-labels", required=False, type=int, default=2, help="The number of labels in the spatial relation prediction task")
	args = parser.parse_args ()
	return args

def main (args):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	srp  = SpatialRelationPrediction (model_name=args.pretrained_model_name,
									  bert_dims=768, 
									  n_labels=args.num_labels,
									  device=device,
									  lr=1e-6)
	if args.test_ids_file == "":
		srp.load_training_data (args.training_data_file, training_frac=args.training_frac)
	else:
		srp.load_training_data (args.training_data_file, test_ids_file=args.test_ids_file)
	srp.start_training (num_epochs=args.num_epochs, context_field=args.context_field, verbose=True)
	srp.save_model (args.model_path)

if __name__ == "__main__":
	main (readArgs ())
