import argparse
import os
import pandas as pd
from collections import Counter

def readArgs ():
    parser = argparse.ArgumentParser (description="Get the most central characters from the books")
    parser.add_argument ("--content-dirs", type=str, required=True, nargs="+", help="directories contain the body of the novels")
    parser.add_argument ("--book-id", type=str, required=True, help="ID of the book in the testset")
    parser.add_argument ("--output-dir", type=str, required=True, help="Directory to store the output files")
    args = parser.parse_args ()
    return args

def booknlp_entities_file_reader (filename, sep='\t'):
    with open (filename) as fin:
        for i, line in enumerate (fin):
            parts = line.strip().split ()
            if i == 0:
                header = parts
                continue
            yield {col: parts[j] for j, col in enumerate (header)}

def main (args):
    found = False
    for dirname in args.content_dirs:
        filename = os.path.join (dirname, f"{args.book_id}.entities")
        if os.path.exists (filename) and os.path.isfile (filename):
            found = not found
            break

    os.makedirs (args.output_dir, exist_ok=True)
    if found:
        chars = list ()
        for row in booknlp_entities_file_reader (filename):
            if row["cat"] == "PER":
                coref = row["COREF"]
                chars.append (coref)
        c = Counter (chars)

        with open (os.path.join (args.output_dir, f"{args.book_id}.per_frequency"), "w") as fout:
            fout.write (f"COREF\tNumTimes\n")
            for key, num in c.most_common ():
                fout.write (f"{key}\t{num}\n")

    else:
        print (f"The book {args.book_id} is not present")


if __name__ == "__main__":
	main (readArgs ())
