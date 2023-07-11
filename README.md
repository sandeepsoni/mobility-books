# Mobility of characters in literature
This project introduces a new task, that of grounding a character to a place in an excerpt from a book.

It's organization is still a work in progress. If you end up using the code or data from this repository, please cite our paper.

Directories
===========

* `modules/`: Directory contains all the reusable code.
* `scripts/`: Directory contains scripts to create annotation examples, learning, and analysis code.
* `data/` : Directory contains the data including annotations and the BERT classifier's predictions.
* `shell_scripts/`: Directory contains wrapper shell scripts.
* `notebooks/`: Directory contains any python notebooks that were used in the process.

Resources
========

[ArXiv](https://arxiv.org/abs/2305.17561)
[Annotation guidelines](https://docs.google.com/document/d/e/2PACX-1vSbMOWfS7gZhGcfQTSdulFsZRxbsrhxFti2jh3Mxc-CkF8LV3qIDWp9VwXZ1vk6svHfu_sEF_F2mV3R/pub)

Citation
========
```
@inproceedings{soni-etal-2023-grounding,
    title = "Grounding Characters and Places in Narrative Text",
    author = "Soni, Sandeep  and
      Sihra, Amanpreet  and
      Evans, Elizabeth  and
      Wilkens, Matthew  and
      Bamman, David",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.655",
    pages = "11723--11736",
    abstract = "Tracking characters and locations throughout a story can help improve the understanding of its plot structure. Prior research has analyzed characters and locations from text independently without grounding characters to their locations in narrative time. Here, we address this gap by proposing a new spatial relationship categorization task. The objective of the task is to assign a spatial relationship category for every character and location co-mention within a window of text, taking into consideration linguistic context, narrative tense, and temporal scope. To this end, we annotate spatial relationships in approximately 2500 book excerpts and train a model using contextual embeddings as features to predict these relationships. When applied to a set of books, this model allows us to test several hypotheses on mobility and domestic space, revealing that protagonists are more mobile than non-central characters and that women as characters tend to occupy more interior space than men. Overall, our work is the first step towards joint modeling and analysis of characters and places in narrative text.",
}
```

Contact
=======

sandeepsoni at berkeley dot edu


