import json
import sys
import csv
from pycldf import Dataset
from pathlib import Path
from collections import defaultdict


csv.field_size_limit(sys.maxsize)  # this is so very large fields can be read from the csv files!

DATA_DIR = Path(__file__).parent / "clics"
GRAPHS_DIR = Path(__file__).parent / "graphs"


###############
# CLICS4 data #
###############
ds = Dataset.from_metadata(DATA_DIR / "clics4" / "cldf" / "StructureDataset-metadata.json")

# read in colexifications
edgelist = []
concept_ids = defaultdict(lambda: len(concept_ids))  # assigns new IDs automatically
for colex in ds.iter_rows("ParameterTable"):
    i = concept_ids[colex["Source_Concept"]]
    j = concept_ids[colex["Target_Concept"]]
    w = colex["Family_Count"]
    edgelist.append((i, j, w))

# read in concept list
concept_coverages = len(concept_ids) * [0]
for i, concept in enumerate(ds.iter_rows("concepts.csv")):
    gloss = concept["Concepticon_Gloss"]
    if gloss in concept_ids:
        concept_id = concept_ids[gloss]
        concept_coverages[concept_id] = concept["Family_Count"]

print(len(concept_ids), len(concept_coverages))

# write to JSON file
json_dict = {"concept_ids": dict(concept_ids), "concept_coverage": concept_coverages, "edgelist": edgelist}
with open(GRAPHS_DIR / "clics4.json", "w") as f:
    json.dump(json_dict, f, indent=4)

##############
# CLIPS data #
##############
ds = Dataset.from_metadata(DATA_DIR / "clips" / "cldf" / "cldf-metadata.json")

# read in colexifications
edgelist = []
concept_ids = defaultdict(lambda: len(concept_ids))  # assigns new IDs automatically
for colex in ds.iter_rows("parameter_network.csv"):
    i = concept_ids[colex["Source"]]
    j = concept_ids[colex["Target"]]
    w = colex["Families"]
    edgelist.append((i, j, w))

# read in concept list
concept_coverages = len(concept_ids) * [0]
for i, concept in enumerate(ds.iter_rows("ParameterTable")):
    gloss = concept["Name"]
    if gloss in concept_ids:
        concept_id = concept_ids[gloss]
        concept_coverages[concept_id] = concept["Families"]

print(len(concept_ids), len(concept_coverages))

# write to JSON file
json_dict = {"concept_ids": dict(concept_ids), "concept_coverage": concept_coverages, "edgelist": edgelist}
with open(GRAPHS_DIR / "clips.json", "w") as f:
    json.dump(json_dict, f, indent=4)