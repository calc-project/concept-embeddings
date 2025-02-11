import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict
from pyconcepticon import Concepticon


BABYCLICS = "babyclics"

ALL_COLUMNS = ["FullFams", "OverlapFams", "AffixFams"]


def read_babyclics(col):
    # define attribute key: affix colexifications (directed) are stored in 'target_concepts',
    # undirected colexifications are stored in 'linked_concepts'
    attr_key = "target_concepts" if col == "AffixFams" else "linked_concepts"

    # get the graph
    con = Concepticon()
    clist = con.conceptlists["List-2023-1308"]

    idxs = [c.id for c in clist.concepts.values()]
    concepts = [clist.concepts[idx].concepticon_gloss for idx in idxs]

    # automatically assigns the next id when a concept is first queried
    concept_to_id = defaultdict(lambda: len(concept_to_id))
    edgelist = []
    visited = set()

    for (idx, concept) in zip(idxs, concepts):
        # iterate over all links and fill the matrix
        # skip over unconnected nodes
        if not clist.concepts[idx].attributes[attr_key]:
            continue

        # get full colexifications
        colex = [(node["NAME"], node[col]) for node in clist.concepts[idx].attributes[attr_key]
                 if node[col] > 0]

        # skip if there are no full colexifications (so no ID is assigned to unconnected node)
        if not colex:
            continue

        id = concept_to_id[concept]
        for node, weight in colex:
            other_id = concept_to_id[node]
            if other_id not in visited or col == "AffixFams":
                # note that 'AffixFams' is a directed network, where G[i, j] != G[j, i]
                edgelist.append((id, other_id, weight))

        visited.add(id)

    return dict(concept_to_id), edgelist


def main(col):
    if not col:
        for c in ALL_COLUMNS:
            main(c)
        return

    if col not in ALL_COLUMNS:
        raise ValueError(f'No column "{col}" in dataset')

    # here the actual main block begins
    print(f'Extracting graph from "babyclics", weighted by "{col}"...')

    # extract the data
    concept_ids, edgelist = read_babyclics(col)

    extracted_data = {"concept_ids": concept_ids, "edgelist": edgelist}

    # write data to JSON file
    fn = col.lower() + ".json"
    out_fp = Path(__file__).parent / "graphs" / fn

    with open(out_fp, "w") as f:
        json.dump(extracted_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--column", type=str)
    args = parser.parse_args()

    # if dataset is underspecified, run main method for both datasets and all columns
    main(args.column)
