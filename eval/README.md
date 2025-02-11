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
 
## Downloading external data

You will need to download two additional pieces of data to run all scripts: FastText vectors and NoRaRe. Both subdirectories (`data/fasttext` and `data/norare`) contain Makefiles that declare downloading (and clearing) the data. You can either `cd` into the respective subdirectories and run `make` there, or conveniently just run `make` or `make download` from this directory to obtain all relevant data. You can then clear all downloaded data from your disk by running `make clear`.

Note that only the NoRaRe data is needed to run the evaluation scripts, since the relevant FastText vectors have been extracted, preprocessed and stored in files in `data/fasttext`. Since dowloading the FastText vectors takes quite some time, the following command only downloads NoRaRe data:

```bash
# download NoRaRe data
graphembeddings/eval$ make
```

If you want to reproduce the preprocessing steps for the FastText vectors, you can run `make download` instead:

```bash
# download NoRaRe data & FastText vectors
graphembeddings/eval$ make download
```

You can remove the downloaded data from your disk by running:

```bash
# remove external data
graphembeddings/eval$ make clear
```

## Creating preprocessed snapshots of the data

Most subdirectories in `data` contain CSV/TSV files that serve as direct input to the evaluation script. These files represent a version of the raw data that is preprocessed for the purposes of our evaluation. All of those files are created using the Python script located in the same directory.

## Running the evaluation scripts

Assuming you have installed all necessary dependencies (see the general README) and downloaded all necessary data, you can run the evaluation scripts to reproduce our analyses. Note that you might obtain slightly different results, since parts of the evaluation rely on non-deterministic methods.
