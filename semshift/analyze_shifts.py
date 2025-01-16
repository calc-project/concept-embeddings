import matplotlib.pyplot as plt
import numpy as np
import seaborn
import csv
import random
from pathlib import Path
from itertools import combinations


embeddings = {}


with open(Path(__file__).parent.parent / "embeddings" / "2025-01-16" / "prone-32.tsv") as f:
#with open(Path(__file__).parent.parent / "embeddings" / "n2v-cbow.tsv") as f:
    for line in f:
        concept, emb = line.strip().split("\t")
        emb = np.array(eval(emb))
        embeddings[concept] = emb

"""
with open(Path(__file__).parent.parent / "norare" / "english_embeddings.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        embeddings[row["CONCEPTICON_GLOSS"]] = eval(row["EMBEDDING"])
"""

# all pairwise similarities between embeddings, results in a normal distribution
similarities = [(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for (x, y) in combinations(embeddings.values(), 2)]
seaborn.displot(similarities, kind="kde", color="b")

# read in attested shifts and calculate pairwise similarities between them
shifts = []
header = True

with open(Path(__file__).parent / "shift_summary.tsv") as f:
    for line in f:
        if header:
            header = False
            continue
        fields = line.strip().split("\t")
        source, target = fields[0], fields[1]
        if source in embeddings and target in embeddings:
            shifts.append((fields[0], fields[1]))

# compare observed shifts with random shifts (where the target is replaced)
correct = 0
total = 0

for source, target in shifts:
    random_target = random.choice(list(embeddings.keys()))
    source_embedding = embeddings[source]
    target_embedding = embeddings[target]
    random_target_embedding = embeddings[random_target]
    # compare cosine similarities
    true_sim = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
    random_sim = np.dot(source_embedding, random_target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(random_target_embedding))
    if true_sim > random_sim:
        correct += 1
    total += 1

print(f"RANDOM TARGETS: {correct}/{total} ({100*(correct/total):.2f}%)")

# compare observed shifts with random shifts (where the source is replaced)
correct = 0
total = 0

for source, target in shifts:
    random_source = random.choice(list(embeddings.keys()))
    source_embedding = embeddings[source]
    target_embedding = embeddings[target]
    random_source_embedding = embeddings[random_source]
    # compare cosine similarities
    true_sim = np.dot(source_embedding, target_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
    random_sim = np.dot(random_source_embedding, target_embedding) / (
                np.linalg.norm(random_source_embedding) * np.linalg.norm(target_embedding))
    if true_sim > random_sim:
        correct += 1
    total += 1

print(f"RANDOM SOURCES: {correct}/{total} ({100 * (correct / total):.2f}%)")

shift_embeddings = [(embeddings[x], embeddings[y]) for x, y in shifts if x in embeddings and y in embeddings]
shift_similarities = [(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in shift_embeddings]
seaborn.displot(shift_similarities, kind="kde", color="r")

plt.show()

print(f"Total concept pairs: {len(shifts)}")
print(f"Concept pairs with valid embeddings: {len(shift_embeddings)}")
print(f"Concept pairs with missing embeddings: {len(shifts) - len(shift_embeddings)}")

