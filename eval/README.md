# Evaluation

This directory contains all materials required to replicate the evaluations and visualizations in the paper. The directory is structured as follows:
* `data` contains all the evaluation data. Each subdirectory contains either a Python script or a Makefile for accessing and processing the original data. **Note that you will have to download some additional data** by running the Makefiles before running the evaluation scripts.
* `figures` contains all figures.
* `results` contains all evaluation metrics for all models.
* `scripts` contains all Python scripts:
  * `msl.py`: Modeling Lexical Semantic Similarity against Multi-SimLex
  * `semshift.py` Predicting Semantic Change against data from DatSemShift
  * `eat.py` Predicting Word Associations against the Edinburgh Assocation Thesaurus
  * `visualize.py` for visualizations
