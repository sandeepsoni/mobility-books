"""
Wrapper script to train a model for spatial relation prediction. 
"""
import argparse
import torch
torch.manual_seed (96)

from sklearn.model_selection import train_test_split
import sys, os

if os.path.abspath ("../") not in sys.path:
    sys.path.append (os.path.abspath ("../"))

from modules.relation_prediction import BERTRelationPrediction
from modules.relation_prediction_constants import * 
import logging

logging.basicConfig (
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

def preprocess_valid_relation_prediction (annotations, *args, **kwargs):
    labels = kwargs.get ("labels", ALL_LABELS)
    full_df = annotations[kwargs.get ("window_size", 10)]
    full_df = full_df.query ("`Spatial Relation` != ''")
    full_df = full_df.query ("`Spatial Relation` in @labels")
    full_df.loc[:,kwargs.get("label_field", "Valid Relation")] = full_df.apply (lambda x: VALID_LABELS[int(x["Spatial Relation"] in SPATIAL_LABELS)], axis=1)

    test_ids_file = kwargs.get ("test_ids_file", "")

    if test_ids_file == "":
        train_df, test_df = train_test_split(full_df, 
                                             test_size=1-kwargs.get ("training_frac", .8), 
                                             random_state=96)
    else:
        with open (test_ids_file) as fin:
            test_ids = [line.strip() for line in fin]
        train_df = full_df.query ("ID not in @test_ids")
        test_df = full_df.query ("ID in @test_ids")

    logging.info (f"All records: {len (full_df)}; for training: {len (train_df)}, for testing: {len (test_df)}")
    return full_df, train_df, test_df

def preprocess_spatial_relation_collapsed_prediction (annotations, *args, **kwargs):
    labels = kwargs.get ("labels", ALL_LABELS)
    full_df = annotations[kwargs.get ("window_size", 10)]
    full_df = full_df.query ("`Spatial Relation` != ''")
    full_df = full_df.query ("`Spatial Relation` in @labels")
    full_df.loc[:,kwargs.get("label_field", "Spatial Relation")] = full_df.apply (lambda x: SPATIAL_RELATION_COLLAPSED_MAP[x["Spatial Relation"]], axis=1)

    test_ids_file = kwargs.get ("test_ids_file", "")
    if test_ids_file == "":
        train_df, test_df = train_test_split(full_df, 
                                             test_size=1-kwargs.get ("training_frac", .8), 
                                             random_state=96)
    else:
        with open (test_ids_file) as fin:
            test_ids = [line.strip() for line in fin]
        train_df = full_df.query ("ID not in @test_ids")
        test_df = full_df.query ("ID in @test_ids")

    logging.info (f"All records: {len (full_df)}; for training: {len (train_df)}, for testing: {len (test_df)}")
    return full_df, train_df, test_df        

def readArgs ():
    parser = argparse.ArgumentParser (description="Script to train and evaluate a spatial relation prediction model")
    parser.add_argument ("--task-name", required=False, choices={"validity", "spatial-collapsed"}, default="validity", type=str, help="Name of the task")
    parser.add_argument ("--pretrained-model-name", required=False, default= "bert-base-cased", type=str, help="Name of the pretrained model")
    parser.add_argument ("--dims", required=False, type=int, default=768, help="Size of contextual embedding")
    parser.add_argument ("--annotated-data-file", required=True, type=str, help="Annotated data is in this pickle file")
    parser.add_argument ("--test-ids-file", required=False, default="", type=str, help="Test file contains IDS on which we want to test")
    parser.add_argument ("--training-frac", required=False, default=0.8, type=float, help="Training fraction")
    parser.add_argument ("--num-epochs", required=False, default=10, type=int, help="Number of epochs for training")
    parser.add_argument ("--num-hidden", required=False, default=0, type=int, help="Number of hidden layers in the network")
    parser.add_argument ("--window-size", required=False, default=100, type=int, help="Specify the size of the window in number of tokens")
    parser.add_argument ("--text-field", required=False, default="context_100", type=str, help="Column name that contains the entire text")
    parser.add_argument ("--model-path", required=True, type=str, help="Path to the file that will store the model")
    parser.add_argument ("--predictions-path", required=False, type=str, default="", help="Path to the file that will store the predictions")
    args = parser.parse_args ()
    return args

def init_config ():
    config_options = {task: {} for task in TASKS}
    config_options["validity"]["preproc_callback"] = preprocess_valid_relation_prediction
    config_options["validity"]["num_labels"] = len (VALID_LABELS)
    config_options["validity"]["label_space"] = VALID_LABELS
    config_options["validity"]["label_field"] = "Valid Relation"

    return config_options

def main (args):
    # Layout the configurations
    config_options = init_config ()

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train and test relation predictor
    predictor = BERTRelationPrediction (model_name=args.pretrained_model_name,
                                        dims=args.dims,
                                        n_labels=config_options[args.task_name]["num_labels"],
                                        n_hidden=args.num_hidden,
                                        device=device,
                                        lr=1e-6,
                                        labels=config_options[args.task_name]["label_space"])
    
    predictor.load_data (args.annotated_data_file, 
                         preprocess=config_options[args.task_name]["preproc_callback"], 
                         test_ids_file=args.test_ids_file, 
                         training_frac=args.training_frac,
                         label_field=config_options[args.task_name]["label_field"],
                         window_size=args.window_size)
	
    predictor.start_training (num_epochs=args.num_epochs, 
                              text_field=args.text_field,
                              label_field=config_options[args.task_name]["label_field"], 
                              verbose=True)
	
    predictor.save (args.model_path,
                    args.predictions_path)

if __name__ == "__main__":
    main (readArgs ())
