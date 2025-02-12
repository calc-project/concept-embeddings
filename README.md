# README

This repository contains all the code and reference points to all the data necessary for reproducing our study *Partial Colexifications Improve Concept Embeddings*. The repository is structured in the following way:
* `data` contains the colexification networks as JSON files, as well as a Python script for retrieving these data using `pyconcepticon`.
* `embeddings` contains all the trained embeddings in a verbose JSON format that also stores relevant information about the training procedure.
* `eval` contains all evaluation scripts and references to the evaluation data.
* `src/graphembeddings` contains all the source code for training different graph embedding models.

## Workflow

To replicate the full workflow of our study, follow these steps:

### 1. Install dependencies

Build the source code and install all necessary dependencies by running:

```
pip install -e .[eval]
```

If you are only interested in training graph embedding models, it is sufficient to run:

```
pip install -e .
```

**Warning: Due to dependency issues, this project can currently not be run in Python >=3.13!**

### 2. Extract the graph data

The files in `data/graphs` were created using the `data/create_graph.py` script, accessing the CLLD Concepticon via its Python API `pyconcepticon`. Simply running the Python script will create the JSON files. Note that you might have to download the [Concepticon data](https://github.com/concepticon/concepticon-data) manually -- in this case, you will need to pass the data filepath explicitly (l.20 in the Python script).

### 3. Train the models

`run.py` is the master script that trains all graph embedding models discussed in the study.

### 4. Fuse embeddings

`fuse.py` fuses embeddings to combine information from different colexification graphs.

### 5. Evaluation

`eval` contains all materials required for the evaluation and visualization discussed in the paper. For the scripts to work, some external data must be downloaded first. For this, see the [README](/eval/README.md) contained directly within the directory.
