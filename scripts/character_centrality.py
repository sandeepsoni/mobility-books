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

def main (args):
    found = False
    for dirname in args.content_dirs:
        filename = os.path.join (dirname, f"{args.book_id}.entities")
        if os.path.exists (filename) and os.path.isfile (filename):
            found = not found
            break

    os.makedirs (args.output_dir, exist_ok=True)
    if found:
        rows = list ()
        df = pd.read_csv (filename, sep="\t")
        df = df[df["cat"] == "PER"]
        for i, row in df.iterrows ():
            coref = row["COREF"]
            cat = row["cat"]
            rows.append (coref)
        c = Counter (rows)

        with open (os.path.join (args.output_dir, f"{args.book_id}.per_frequency"), "w") as fout:
            fout.write (f"COREF\tNumTimes\n")
            for key, num in c.most_common ():
                fout.write (f"{key}\t{num}\n")

    else:
        print ("The book is not present")


if __name__ == "__main__":
	main (readArgs ())
