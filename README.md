# Mobility of characters in fiction

This project aims to detect and reason about the mobility of characters in fiction.

Directories
===========

* `data/`: This directory contains the data
  * `v1`: `.collocations` file contain PER and GPE entities that are PROP tagged.
  * `v2`: `.collocations` file contains PER and (GPE, FAC, LOC) entities that are tagged in any way.

* `scripts/`: This directory contains the python scripts to different tasks
  * `name_location_cooccurrences.py`: This script takes one book as input and outputs a 
                                      character name X location collocations.

* `shell_scripts/`: This directory contains the shell scripts that execute the python scripts.
  * `make_character_locations.sh`: This script executes the `name_location_cooccurrences.py` 
                                   for all books as input.
