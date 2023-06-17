""" This script is used to run BookNLP pipeline on a book"""

from booknlp.booknlp import BookNLP
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser (description="Use BookNLP on entire collection of books")
parser.add_argument ("--input-dir", type=str, required=True, help="Directory contains the input directory")
parser.add_argument ("--bookids-file", type=str, required=True, help="File contains book ids")
parser.add_argument ("--output-dir", type=str, required=True, help="Output directory will create the files")
args = parser.parse_args ()

def output_files_exist (output_dir, book_id):
    return all ([
        os.path.exists (os.path.join (output_dir, f"{book_id}.book")),
        os.path.exists (os.path.join (output_dir, f"{book_id}.book.html")),
        os.path.exists (os.path.join (output_dir, f"{book_id}.tokens")),
        os.path.exists (os.path.join (output_dir, f"{book_id}.entities")),
        os.path.exists (os.path.join (output_dir, f"{book_id}.quotes")),
        os.path.exists (os.path.join (output_dir, f"{book_id}.supersense"))

    ])

# Read the book_ids from file
with open (args.bookids_file) as fin:
    book_ids = fin.read().splitlines ()

os.makedirs (args.output_dir, exist_ok=True)

# Configure BookNLP
model_params={
    "pipeline":"entity,quote,supersense,event,coref",
    "model":"big"
}

booknlp=BookNLP("en", model_params)

for book_id in tqdm (book_ids):
    input_filename = os.path.join (args.input_dir, f"{book_id}.txt")
    if output_files_exist (args.output_dir, book_id):
        continue
    booknlp.process(input_filename, args.output_dir, book_id)