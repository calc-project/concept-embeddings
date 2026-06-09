import csv
import random
import numpy as np
from collections import defaultdict
from pathlib import Path
from graphembeddings.eval.semshift import cosine_similarity
from graphembeddings.utils.io import read_embeddings
from itertools import permutations, combinations


analogies_file = Path(__file__).parent.parent / "data" / "bats" / "analogies.tsv"
analogies = defaultdict(list)
symmetric_categories = ['L09 [antonyms - gradable]', 'L10 [antonyms - binary]']

with open(analogies_file) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["CONCEPT_1"] and row["CONCEPT_2"]:
            source_concept = row["CONCEPT_1"]
            target_concepts = row["CONCEPT_2"].split("/")
            category = row["BATS_CATEGORY"]
            analogies[category].append((source_concept, target_concepts))
            if category in symmetric_categories:
                for target_concept in target_concepts:
                    analogies[category].append((target_concept, [source_concept]))

male_female_analogies = list(analogies['E10 [male - female]'])


def sample_analogy(analogies):
    # analogies should be a list of pairs
    s1, t1 = s2, t2 = random.choice(analogies)
    while t1 == t2:
        s2, t2 = random.choice(analogies)

    return s1, t1, s2, t2


def create_analogies(analogies, embeddings):
    result = []

    for pair1, pair2 in permutations(analogies, 2):
        s1, t1_list = pair1
        s2, t2 = pair2
        for t1 in t1_list:
            t2 = [concept for concept in t2 if concept in embeddings and concept != t1]
            if t2 and {s1, t1, s2}.issubset(embeddings.keys()):
                result.append((s1, t1, s2, t2))

    return result


def vector_offset_closest_neighbor(c1, c2, c3, embeddings):
    emb1 = np.array(embeddings[c1])
    emb2 = np.array(embeddings[c2])
    emb3 = np.array(embeddings[c3])
    # king - man + woman = queen
    offset_vector = emb2 - emb1 + emb3

    best_match = None
    best_similarity = -1
    for concept, vector in embeddings.items():
        if concept in [c1, c2, c3]:
            continue
        vector = np.array(vector)
        sim = cosine_similarity(offset_vector, vector)
        if sim > best_similarity:
            best_match = concept
            best_similarity = sim

    return best_match

#embeddings_file = Path(__file__).parent.parent.parent / "embeddings" / "full-affix" / "prone.json"
#embeddings_file = Path(__file__).parent.parent.parent / "output" / "semantic-node2vec-sbert-full-affix-overlap.json"
embeddings_file = Path(__file__).parent.parent.parent / "output" / "fullfams" / "semantic-node2vec-2.json"
print(embeddings_file)
embeddings = read_embeddings(embeddings_file)

# print(vector_offset_closest_neighbor("FATHER", "MOTHER", "BULL", embeddings))

for category, pairs in analogies.items():
    analogies_per_category = create_analogies(pairs, embeddings)
    total = len(analogies_per_category)
    if total < 100:
        continue
    matches = 0
    for (s1, t1, s2, t2) in analogies_per_category:
        pred = vector_offset_closest_neighbor(s1, t1, s2, embeddings)
        if pred in t2:
            matches += 1
    print(f"{category}: {matches} / {total} ({matches / total:.4f})")
