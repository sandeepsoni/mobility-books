#! /bin/bash

parallel --jobs 16 --tmpdir ~/tmp/ python ../scripts/third_person_travel_books.py --input-filename {} --output-dir ../data/third-person-narrative ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_1_4/*.tokens)

parallel --jobs 16 --tmpdir ~/tmp/ python ../scripts/third_person_travel_books.py --input-filename {} --output-dir ../data/third-person-narrative ::: $(ls /mnt/data0/kentkchang/charemotions/corpus/booknlp.1.0.7/gutenberg_fiction_tagged_5_9/*.tokens)

