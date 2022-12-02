import argparse
import pandas as pd

def readArgs ():
    parser = argparse.ArgumentParser (description="Aggregate all the geonames records")
    parser.add_argument ("--input-file", type=str, required=True, help="Input file contains geonames file")
    parser.add_argument ("--output-file", type=str, required=True, help="Output file contains only the place names and their geo coordinates")
    args = parser.parse_args ()
    return args

def main (args):
    rows = list ()
    with open (args.input_file) as fin:
        for line in fin:
            parts = line.strip().split ("\t")
            geo_id, name, lat, lon = parts[0], parts[2], parts[4], parts[5]
            rows.append ([geo_id, name, lat, lon])

    df = pd.DataFrame (rows, columns=["geo_id", "name", "lat", "lon"])
    df.to_csv (args.output_file, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main (readArgs ())
