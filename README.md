# README

To replicate the full workflow of our study, follow these steps:

## 1. Install dependencies

Build the source code and install all necessary dependencies by running:

```
pip install -e .[eval]
```

If you are only interested in training the concept embeddings, it is sufficient to run:

```
pip install -e .
```

## 2. Clone Concepticon data

The training procedure assumes that you have a copy of the Concepticon data at the root of this repository. Clone the data first by running:

```
git clone --depth 1 --branch v3.4.0 https://github.com/concepticon/concepticon-data
```

## 3. Extract the graph data (optional)

The files in `data/graphs` were created using the scripts `data/create_graph.py` (colexifications from IDS) and `data/create_graph_clics.py (CLICS4, CLIPS), accessing the CLLD Concepticon via its Python API `pyconcepticon`. Simply running the Python script will create the JSON files.

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
