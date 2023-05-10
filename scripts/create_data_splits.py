"""
Create the train/dev/test splits from the overall data.

python create_data_splits.py \
    --annotation-files ../data/annotations/examples_to_annotate.v1.tsv ../data/annotations/examples_to_annotate.v2.tsv \
    --train-file ../data/annotations/train.tsv \
    --dev-file ../data/annotations/dev.tsv \
    --test-file ../data/annotations/test.tsv \
    --train-frac 0.7 \
    --dev-frac 0.1 \
    --random-state 96

"""

import argparse
import pandas as pd

def split_df(df, train_frac=0.7, dev_frac=0.1, random_state=None):
    """
    Code generated through BARD and modified slightly
    Splits a Pandas DataFrame into three: training_df, dev_df, and test_df.
    
    Args:
    df: The Pandas DataFrame to split.
    train_frac: The fraction of the DataFrame to use for the training set.
    dev_frac: The fraction of the DataFrame to use for the development set.
    random_state: The random seed to use for shuffling the data and splitting the DataFrame.

    Returns:
    A tuple of three Pandas DataFrames: training_df, dev_df, and test_df.
    """

    # Shuffle the DataFrame.
    df = df.sample(frac=1, random_state=random_state)

    # Split the DataFrame into three.
    training_df = df.iloc[:int(len(df) * train_frac)]
    dev_df = df.iloc[int(len(df) * train_frac):int(len(df) * (train_frac + dev_frac))]
    test_df = df.iloc[int(len(df) * (train_frac + dev_frac)):]

    return training_df, dev_df, test_df

def readArgs ():
    parser = argparse.ArgumentParser (description="Create the train/dev/test splits from the data")
    parser.add_argument ("--annotation-files", type=str, required=True, nargs="+", help="Files that contain the annotation examples")
    parser.add_argument ("--train-file", type=str, required=True, help="File will contain examples used for training")
    parser.add_argument ("--dev-file", type=str, required=True, help="File will contain examples in the dev set (used for hyperparameter tuning)")
    parser.add_argument ("--test-file", type=str, required=True, help="File will contain the examples in the test set")
    parser.add_argument ("--train-frac", type=float, required=False, default=0.7, help="The fraction of examples that are used for training")
    parser.add_argument ("--dev-frac", type=float, required=False, default=0.1, help="The fraction of examples that are used for development")
    parser.add_argument ("--random-state", type=int, required=False, default=96, help="The random seed used to initialize the splits")
    return parser.parse_args ()

def main (args):
    df = pd.concat([pd.read_csv (filename, sep='\t') for filename in args.annotation_files])
    training_df, dev_df, test_df = split_df(df, train_frac=args.train_frac, dev_frac=args.dev_frac, random_state=args.random_state)
    training_df.to_csv (args.train_file, sep='\t', header=True, index=False)
    dev_df.to_csv (args.dev_file, sep='\t', header=True, index=False)
    test_df.to_csv (args.test_file, sep='\t', header=True, index=False)

if __name__ == "__main__":
    main (readArgs ())