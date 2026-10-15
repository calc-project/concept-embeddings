# README

This repository contains the code for training and inferring the concept embeddings described in this study:

> Rubehn, A. (2026). Inductive Learning of Concept Embeddings from Cross-Linguistic Colexification Patterns with Textual Attributes. arXiv preprint.

If you use this package in your work, please cite the work above accordingly.

*Note: If you are interested in the code behind the paper by [Rubehn and List (2025)](https://aclanthology.org/2025.acl-long.1004/), please check out v1.0 of this repository.*

## Quickstart

This package can be installed from PyPi using:

```
pip install conceptembeddings
```

The package includes a wrapper for the pretrained model. It requires that Concepticon data are cloned on your device.

```python
from conceptembeddings import ConceptEmbeddings
from pyconcepticon import Concepticon

concepticon = Concepticon("path/to/concepticon-data")
ce = ConceptEmbeddings(concepticon=concepticon)

# embed a concept defined in Concepticon
emb_water = ce("WATER")

# embed a concept with a custom definition
emb_orc = ce.embed_definition("A fictional, aggressive humanoid creature common in fantasy literature and games")
```

If you have set up a "global" instance of Concepticon with `cldfbench`, you can instantiate a `ConceptEmbeddings` object without passing a `Concepticon` instance:

```python
ce = ConceptEmbeddings()  # tries to access a globally defined Concepticon instance
```

## Study Replication

To replicate the full workflow of our study, follow these steps:

### 1. Install dependencies

Build the source code and install all necessary dependencies by running:

```
pip install -e .[eval]
```

If you are only interested in training the concept embeddings, it is sufficient to run:

```
pip install -e .
```

### 2. Clone Concepticon data

The training procedure assumes that you have a copy of the Concepticon data at the root of this repository. Clone the data first by running:

```
git clone --depth 1 --branch v3.4.0 https://github.com/concepticon/concepticon-data
```

### 3. Extract the graph data (optional)

The files in `data/graphs` were created using the scripts `data/create_graph.py` (colexifications from IDS) and `data/create_graph_clics.py` (CLICS4, CLIPS), accessing the CLLD Concepticon via its Python API `pyconcepticon`. Simply running the Python script will create the JSON files.

### 4. Train the model

Train the model by running:

```
python run.py --config config.yaml
```

Hyperparameters are set as described in the study; you can adjust them by modifying `config.yaml`.

### 5. Evaluation

`eval` contains all materials required for the evaluation and visualization discussed in the paper. Some scripts rely on data from NoRaRe which must be downloaded first. For this, run:

```
cd eval/data/norare && make
```
