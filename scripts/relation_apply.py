import os, sys
import argparse
import pandas as pd
import torch

if os.path.abspath ("../") not in sys.path:
    sys.path.append (os.path.abspath ("../"))
from modules.relation_prediction import BERTRelationPrediction
from modules.relation_prediction_constants import * 
import logging

logging.basicConfig (
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

def readArgs ():
    parser = argparse.ArgumentParser (description="Apply a trained model to each book individually")
    parser.add_argument ("--validity-model-path", type=str, required=True, help="Path to the model file that contains the validity model")
    #parser.add_argument ("--spatial-model-path", type=str, required=True, help="Path to the model file that contains the spatial model")
    #parser.add_argument ("--temporal-span-model-path", type=str, required=True, help="Path to the model file that contains the temporal_span model")
    #parser.add_argument ("--narrative-tense-model-path", type=str, required=True, help="Path to the model file that contains the narrative_tense model")
    parser.add_argument ("--book-ids", type=str, required=True, nargs="+", help="IDs of the book")
    parser.add_argument ("--collocations-dir", type=str, required=True, help="Path to the corpus containing collocations")
    parser.add_argument ("--output-dir", type=str, required=True, help="Path to the output file")

    return parser.parse_args ()

def init_config ():
    config_options = {task: {} for task in TASKS}
    config_options["validity"]["num_labels"] = len (VALID_LABELS)
    config_options["validity"]["label_space"] = VALID_LABELS
    config_options["validity"]["label_field"] = "valid_relation"

    config_options["spatial"]["num_labels"] = len (SPATIAL_RELATION_LABELS)
    config_options["spatial"]["label_space"] = SPATIAL_RELATION_LABELS
    config_options["spatial"]["label_field"] = "spatial_relation"

    config_options["temporal_span"]["num_labels"] = len (TEMPORAL_SPAN_LABELS)
    config_options["temporal_span"]["label_space"] = TEMPORAL_SPAN_LABELS
    config_options["temporal_span"]["label_field"] = "temporal_span"

    config_options["narrative_tense"]["num_labels"] = len (NARRATIVE_TENSE_LABELS)
    config_options["narrative_tense"]["label_space"] = NARRATIVE_TENSE_LABELS
    config_options["narrative_tense"]["label_field"] = "narrative_tense"

    return config_options
    
def main (args):
    config_options = init_config()
    # load the validity checkpoint
    validity_checkpoint = torch.load (args.validity_model_path)
    validity_model = BERTRelationPrediction(n_labels=config_options["validity"]["num_labels"],
                                            labels=config_options["validity"]["label_space"])
    validity_model.load_state_dict (validity_checkpoint["model_state_dict"])

    # load the spatial model
    #spatial_checkpoint = torch.load (args.spatial_model_path)
    #spatial_model = BERTRelationPrediction (n_labels=config_options["spatial"]["num_labels"],
    #                                        labels=config_options["spatial"]["label_space"])
    #spatial_model.load_state_dict (spatial_checkpoint["model_state_dict"])

    # load the temporal span model
    #temporal_span_checkpoint = torch.load (args.temporal_span_model_path)
    #temporal_span_model = BERTRelationPrediction (n_labels=config_options["temporal_span"]["num_labels"],
    #                                              labels=config_options["temporal_span"]["label_space"])
    #temporal_span_model.load_state_dict (temporal_span_checkpoint["model_state_dict"])

    # load the narrative tense model
    #narrative_tense_checkpoint = torch.load (args.narrative_tense_model_path)
    #narrative_tense_model = BERTRelationPrediction (n_labels=config_options["temporal_span"]["num_labels"],
    #                                                labels=config_options["temporal_span"]["label_space"])
    #narrative_tense_model.load_state_dict (narrative_tense_checkpoint["model_state_dict"])
        
    for book_id in args.book_ids:
        book_df = pd.read_csv (os.path.join (args.collocations_dir, f"{book_id}.examples"), sep="\t")
        validity_predictions = validity_model.apply_book (book_df,
                                                             text_field=config_options["validity"]["label_field"])
        predictions = [VALID_LABELS[prediction] for prediction in predictions]
        book_df["binary_classifier_predictions"] = predictions
        book_df = book_df[book_df["binary_classifier_predictions"] == "VALID"]
        os.makedirs (args.output_dir, exist_ok=True)
        book_df.to_csv (os.path.join (args.output_dir, f"{book_id}.predictions"), sep="\t", header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())