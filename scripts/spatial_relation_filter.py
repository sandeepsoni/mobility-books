""" Distinguish between bad extraction and valid extraction examples
"""

import argparse
import torch
torch.manual_seed (96)
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json

from transformers import BertTokenizer, BertModel
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from tqdm import tqdm

class BERTRelationPrediction (nn.Module):
    def __init__ (self, model_name="bert-base-cased", bert_dims=768, n_labels=8):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False, do_basic_tokenize=False)
        self.bert = BertModel.from_pretrained(model_name)
        self.n_labels = n_labels
        self.fc = nn.Linear (2*bert_dims, self.n_labels)
        
    def forward (self, text, per_entity_span, loc_entity_span, device="cpu"):
        # get entity span representations, concatenate and pass it through a 
        # feedforward network.
        token_wordpieces = self.tokenizer.convert_ids_to_tokens (self.tokenizer (text)['input_ids'][1:-1])
        per_entity_wordpieces = self.tokenizer.convert_ids_to_tokens(self.tokenizer (per_entity_span)['input_ids'][1:-1])
        loc_entity_wordpieces = self.tokenizer.convert_ids_to_tokens(self.tokenizer (loc_entity_span)['input_ids'][1:-1])
        
        per_entity_positions = search_sublist (token_wordpieces, per_entity_wordpieces)
        loc_entity_positions = search_sublist (token_wordpieces, loc_entity_wordpieces)
        
        encoded_input = self.tokenizer (text, return_tensors="pt")
        encoded_input.to(device)
        _, pooled_inputs, sequence_outputs =  self.bert (**encoded_input, output_hidden_states=True, return_dict=False)
        last_layer_output = sequence_outputs[-1][0]
        per_entity_repr = last_layer_output[per_entity_positions[0][0]: per_entity_positions[0][1],:].mean (dim=0)
        loc_entity_repr = last_layer_output[loc_entity_positions[0][0]: loc_entity_positions[0][1],:].mean (dim=0)
        
        input_repr = torch.cat ((per_entity_repr, loc_entity_repr), 0)
        output = self.fc (input_repr)
        return output
    
    def evaluate (self):
        pass

def readArgs ():
    parser = argparse.ArgumentParser (description="Script to run a baseline BERT model")
    parser.add_argument ("--input-filename", type=str, required=True, help="File contains the annotated data")
    parser.add_argument ("--pretrained-model-name", type=str, required=False, default="bert-base-cased", help="Name of the pretrained model")
    parser.add_argument ("--num-training-examples", type=int, required=False, default=1000, help="Number of training examples to consider in training")
    parser.add_argument ("--num-epochs", type=int, required=False, default=10, help="Number of epochs used in training")
    parser.add_argument ("--context", type=str, required=False, default="context_10", help="The amount of linguistic context for making predictions")
    parser.add_argument ("--output-filename", type=str, required=True, help="File contains the output data")
    args = parser.parse_args ()
    return args


def search_sublist (sentence_as_tokens, entity_as_tokens):
    """
    For the entity span (decomposed into tokens), find all the start and end
    positions within the sentence (decomposed into tokens)
    """
    sublist_positions = list ()
    for i, token in enumerate (sentence_as_tokens):
        #print (token, entity_as_tokens[0])
        if token == entity_as_tokens[0]: # see if the first character matches
            if all ([tok == sentence_as_tokens[i+j] if (i+j) < len(sentence_as_tokens) else False for j,tok in enumerate (entity_as_tokens)]):
                sublist_positions.append ((i, i+len(entity_as_tokens)))
                
    return sublist_positions

def main (args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_labels = [ "NO RELATIONSHIP ASSERTED",
                   "TOWARD(got there)",
                   "FROM",
                   "NEAR",
                   "IN",
                   "NEGATIVE ASSERTION",
                   "THROUGH",
                   "TOWARD (uncertain got there)",
                   "BAD LOC",
                   "BAD PER",
                   "UNCERTAIN ASSERTION"]

    bad_labels = [ "BAD LOC",
                   "BAD PER",
                   "UNCERTAIN ASSERTION"]

    label_names = {0:"GOOD", 1:"BAD"}
    accepted_labels = ["GOOD", "BAD"]

    df = pd.read_csv (args.input_filename, sep="\t")
    df = df[df["Spatial Relation"] != ""]
    df = df[df["Spatial Relation"].isin (all_labels)]
    df["Spatial SuperRelation"] = df["Spatial Relation"].isin (bad_labels)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=96)

    bertRE = BERTRelationPrediction (model_name=args.pretrained_model_name, bert_dims=768, n_labels=2)
    bertRE.to(device)
    optimizer = torch.optim.Adam(bertRE.parameters(), lr=1e-5)
    cross_entropy=nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        print (f"Epoch: {epoch}")
        bertRE.train()
        # Train
        for i in tqdm (range (len (train_df[0:args.num_training_examples]))):
            # get the extracted quantities
            text = train_df[args.context].iloc[i]
            per_entity_span = train_df["persons_text"].iloc[i]
            loc_entity_span = train_df["locations_text"].iloc[i]
            label = label_names[int (train_df["Spatial SuperRelation"].iloc[i])]
            #label = train_df["Spatial Relation"].iloc[i] 
            y_pred = bertRE.forward (text, per_entity_span, loc_entity_span, device=device)
            y_truth = accepted_labels.index (label)
            #print (y_pred, torch.tensor (y_truth))
            loss = cross_entropy (y_pred.unsqueeze (0), torch.tensor ([y_truth]).to(device))
            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()
            
            
        groundtruth, predictions = list (), list ()
        bertRE.eval()
        with torch.no_grad():
            for i in tqdm (range (len (test_df))):
                # get the extracted quantities
                text = test_df[args.context].iloc[i]
                per_entity_span = test_df["persons_text"].iloc[i]
                loc_entity_span = test_df["locations_text"].iloc[i]
                #label = test_df["Spatial Relation"].iloc[i]
                label = label_names[int (test_df["Spatial SuperRelation"].iloc[i])]
                y_truth = accepted_labels.index (label)
                y_pred = bertRE.forward (text, per_entity_span, loc_entity_span, device=device)
                groundtruth.append (y_truth)
                predictions.append (torch.argmax (torch.nn.functional.softmax (y_pred)).item())

        print (classification_report (groundtruth, predictions))
        print (classification_report (groundtruth, [4]*len(predictions))) #baseline

    test_df["ground_truth"] = groundtruth
    test_df["predictions"] = predictions

    test_df.to_csv (args.output_filename, sep="\t", index=False, header=True)
    #out_df = pd.DataFrame.from_dict ({"ground_truth": groundtruth, "predictions": predictions})
    #out_df.to_csv ("model_output.tsv")

if __name__ == "__main__":
    main (readArgs ())
