import matplotlib.pyplot as plt
import numpy as np
import seaborn
import csv
import random
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LogisticRegression

from graphembeddings.utils.io import read_embeddings, read_ft_embeddings
from graphembeddings.utils.postprocess import fuse_embeddings, cosine_similarity


AFFIX_EMB_FP = Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "affixfams" / "prone.json"
FULL_EMB_FP = Path(__file__).parent.parent.parent / "embeddings" / "babyclics" / "fullfams" / "prone.json"

affix_embeddings = read_embeddings(AFFIX_EMB_FP)
full_embeddings = read_embeddings(FULL_EMB_FP)

# embeddings = full_embeddings
# embeddings = fuse_embeddings(affix_embeddings, full_embeddings)

FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"
embeddings = read_ft_embeddings(FT_EMBEDDINGS_DIR / "estonian_embeddings.csv")

# all pairwise similarities between embeddings, results in a normal distribution
similarities = [(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for (x, y) in combinations(embeddings.values(), 2)]
# seaborn.displot(similarities, kind="kde", color="b")

# read in attested shifts and calculate pairwise similarities between them
shifts = []
header = True

with open(Path(__file__).parent.parent / "data" / "semshift" / "shift_summary.tsv") as f:
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

random_shifts = []

for source, target in shifts:
    random_target = random.choice(list(embeddings.keys()))
    random_shifts.append((source, random_target))
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
    random_shifts.append((random_source, target))
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
shift_similarities = [cosine_similarity(x, y) for x, y in shift_embeddings]
# seaborn.displot(shift_similarities, kind="kde", color="r")

# plt.show()

# logistic regression for observed shifts vs randomly sampled shifts
y = np.array(len(shifts) * [1] + len(random_shifts) * [0])
random_shift_embeddings = [(embeddings[x], embeddings[y]) for x, y in random_shifts]
random_shift_similarities = [cosine_similarity(x, y) for x, y in random_shift_embeddings]
X = np.array(shift_similarities + random_shift_similarities).reshape(-1, 1)

assert len(X) == len(y)

lr = LogisticRegression()
lr.fit(X, y)
print(lr.score(X, y))
