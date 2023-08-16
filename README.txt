Authors
-------

Rom Gat 	(ID: 206395253)
Shiran Naor	(ID: 300212784)


About
-----

This repository serves as a complete solution to the requirements presented in the final project of the NLP course.

This reposirtory contains:

- The folder `resources` which contains various python files:
	- `scv_scrape.py`	Contains code used for scraping data from the supreme court website.
	- `utils.py`		Contains various utility functions for all other files.
	- `ner.py`		Contains usable and helper methods for Named Entity Recognition (Used for sections 2,6,7).
	- `keywords.py`		Contains usable and helper methods for Keyword Extraction (Used for sections 3,4,10).
	- `sentiment.py`	Contains usable and helper methods for Sentiment Analysis (Used for sections 8).

- The folder `corpus` which contains the data.

- The folder `excel` which contains excel files with results for the various tasks.

- The folder `learning_curves` which contains learning curves for the 3 models implemented and trained in this project.

- The Jupyter notebook `NLP_Final_Project_Demo.ipynb` which uses all python files above to generate all results. This version of the notebook was already ran meaning that all results are there to view.


If you wish to run the included notebook, you need to upload it to google colab and run. It already clones the whole repository to temporary memory for use.