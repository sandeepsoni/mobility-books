import argparse
import os

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
		filename = os.path.join (dirname, f"{args.book_id}.tokens")
		if os.path.exists (filename) and os.isfile (filename):
			found = not found
			break

	if found:
		print (filename)

if __name__ == "__main__":
	main (readArgs ())
