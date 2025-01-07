import csv
from collections import defaultdict


# define target columns
threshold = 0.0
weight_criterion = "Family_Count"
target_columns = ["Source_Concept", "Target_Concept", "Family_Count", "Family_Weight"]

data = []

with open("colexifications.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_dict = {c: row[c] for c in target_columns}
        data.append(row_dict)


# create edgelist and ID dict
concept_to_id = defaultdict(lambda: len(concept_to_id))
edgelist = []

for row in data:
    source_concept = row["Source_Concept"]
    target_concept = row["Target_Concept"]
    weight = float(row[weight_criterion])
    if weight > threshold:
        source_id = concept_to_id[source_concept]
        target_id = concept_to_id[target_concept]
        edgelist.append((source_id, target_id, weight))


# write edgelist file
with open(f"edgelist-{weight_criterion}.tsv", "w") as f:
    f.write(f"# {len(concept_to_id)}\n")
    for source, target, weight in edgelist:
        f.write(f"{source}\t{target}\t{weight}\n")

# write ID file
with open(f"concept-ids-{weight_criterion}.tsv", "w") as f:
    for concept, id in concept_to_id.items():
        f.write(f"{id}\t{concept}\n")
