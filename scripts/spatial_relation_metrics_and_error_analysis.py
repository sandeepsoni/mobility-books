import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import classification_report

def readArgs ():
    parser = argparse.ArgumentParser (description="Analyze the errors of spatial prediction")
    parser.add_argument ("--metadata-file", type=str, required=True, help="File contains the metadata")
    parser.add_argument ("--metrics-file", type=str, required=True, help="File contains the metrics")
    args = parser.parse_args ()
    return args

def main (args):
    metadata = pd.read_csv (args.metadata_file, sep="\t", names=["training_size", "context_size", "file_prefix"])
    dirname = os.path.dirname (args.metadata_file)
    rows = list ()
    for i, row in metadata.iterrows ():
        file_prefix = row["file_prefix"]
        filename = os.path.join (dirname, f"{file_prefix}.tsv")
        df = pd.read_csv (filename, sep="\t")
        row["accuracy_score"] = accuracy_score(df["ground_truth"], df["predictions"])
        row["f1_score"] = f1_score (df["ground_truth"], df["predictions"], average="macro", zero_division=0)
        row["precision_score"] = precision_score (df["ground_truth"], df["predictions"], average="macro", zero_division=0)
        row["recall_score"] = recall_score (df["ground_truth"], df["predictions"], average="macro", zero_division=0)
        rows.append (row)

    metadata = pd.DataFrame (rows)
    metadata.to_csv (args.metrics_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
