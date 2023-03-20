""" This script does:

    (a) Train a model for spatial relation prediction
    (b) Load an already trained model and apply it to unseen data.

"""
import argparse
import torch

class SpatialRelationPredictor (object):
    def __init__ (self):
        pass

    def load_training_data (self, ):
        pass

    def train (self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            # Train
            for i in tqdm (range (len (train_df[0:args.num_training_examples]))):
                # get the extracted quantities
                text = train_df[args.context].iloc[i]
                per_entity_span = train_df["persons_text"].iloc[i]
                loc_entity_span = train_df["locations_text"].iloc[i]
                label = train_df["Spatial Relation"].iloc[i] 
                y_pred = bertRE.forward (text, per_entity_span, loc_entity_span, device=device)
                y_truth = accepted_labels.index (label)
                loss = cross_entropy (y_pred.unsqueeze (0), torch.tensor ([y_truth]).to(device))
                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

    def save (self, path):
        pass


def readArgs ():
    parser = argparse.ArgumentParser (description="Train models and apply it to books")
    parser.add_argument ("--mode", required=False, type=str, default="train", choices={"infer", "train"}, help="The mode of operation for the script")
    parser.add_argument ("--model-path", required=True, type=str, help="The model path where the model is saved")
    args = parser.parse_args ()
    return args

def main (args):
    srp = SpatialRelationPredictor ()
    if args.mode == "train":
        srp.load_training_data ()
        srp.train ()
        srp.save (args.model_path)
    else:
        srp.load_model (args.model_path)
        srp.load_prediction_data ()
        srp.predict ()
        srp.save_predictions ()

if __name__ == "__main__":
    main (readArgs ())

