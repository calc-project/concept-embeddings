import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict

from conceptembeddings.pretrained import ConceptEmbeddings


SHIFTS_DEFAULT_FP = Path(__file__).parent.parent / "data" / "semshift" / "shift_summary.tsv"

embeddings = ConceptEmbeddings()
shift_types = defaultdict(lambda: len(shift_types))
X = []
Y = []

with open(SHIFTS_DEFAULT_FP) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        shift_type = row["Shift_Type"]
        if not shift_type:
            continue
        y = shift_types[shift_type]
        x = embeddings(row["Target"]) - embeddings(row["Source"])
        X.append(x)
        Y.append(y)

X = np.array(X)
Y = np.array(Y)

# kmeans = KMeans(n_clusters=6, n_init=5).fit(X)
X_transformed = PCA(n_components=2).fit_transform(X)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=Y, cmap="viridis")
plt.show()

