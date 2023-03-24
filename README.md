# Mobility of characters in literature
This project aims to ground characters in a story to places.

Directories
===========

* `modules/`: Directory contains all the reusable code.
* `scripts/`: Directory contains scripts to create annotation examples, learning, and analysis code.
  * `name_location_cooccurrences.py`: Script takes one book as input and outputs character name X place collocations.
  * `create_annotation_sample.py`: Script takes the collocations and the original content to create an annotation sample.
* `shell_scripts/`: Directory contains wrapper shell scripts.
  * `make_character_locations.sh`: Shell script executes the `name_location_cooccurrences.py` for all books as input.
  * `create_annotation_sample.sh`: Shell script executes the `create_annotation_sample.py` script
