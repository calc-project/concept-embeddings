import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from conceptembeddings.pretrained import ConceptEmbeddings
from pathlib import Path
from graphembeddings.utils.io import read_graph_data


GRAPH_DIR = Path(__file__).parent.parent.parent / "data" / "graphs"
_, _, clics_concepts, _ = read_graph_data(GRAPH_DIR / "clics4.json")
_, _, clips_concepts, _ = read_graph_data(GRAPH_DIR / "clips.json")

transductive_concepts = set(clics_concepts) | set(clips_concepts)


concepts = []
categories = []

with open("Dunabeitia-2025-MultiPic.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["CONCEPTICON_GLOSS"] and row["CATEGORY_CONSENSUS"]:
            concepts.append(row["CONCEPTICON_GLOSS"])
            categories.append(row["CATEGORY_CONSENSUS"])


category_counter = Counter(categories)
filtered_concepts, filtered_categories = [], []
for concept, category in zip(concepts, categories):
    if category_counter[category] > 10:
        filtered_concepts.append(concept)
        filtered_categories.append(category)

categories = filtered_categories
concepts = filtered_concepts

unique_categories = sorted(set(categories))
cmap = cm.get_cmap("nipy_spectral")
# spread the categories evenly over the colormap
category_colors = {
    c: cmap(i / max(len(unique_categories) - 1, 1))
    for i, c in enumerate(unique_categories)
}

ce = ConceptEmbeddings()
embedding_matrix = [ce(c) for c in concepts]
pca_res = PCA(n_components=2).fit_transform(embedding_matrix)

categories = np.asarray(categories)

fig, ax = plt.subplots()
#for c in unique_categories:
#    mask = categories == c
#    ax.scatter(pca_res[mask, 0], pca_res[mask, 1],
#               color=category_colors[c], label=c)

registered_categories = []

for pos, concept, category in sorted(zip(pca_res, concepts, categories), key=lambda x: x[2]):
    color = category_colors[category]
    if category == "buildingsroomsstructures":
        category = "buildings"
    marker = "." if concept in transductive_concepts else "*"
    if marker == "." and category not in registered_categories:
        ax.scatter(pos[0], pos[1], color=color, marker=marker, label=category)
        registered_categories.append(category)
    else:
        ax.scatter(pos[0], pos[1], color=color, marker=marker)

ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
leg = ax.legend(title="Category", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

def widen_for_legend(fig, ax, leg, pad=0.2):
    """Grow the figure to the right so the legend fits, keeping the axes size."""
    fig.canvas.draw()
    dpi = fig.dpi
    old_w, h = fig.get_size_inches()
    pos = ax.get_position()

    # absolute axes geometry, in inches
    ax_x0, ax_w = pos.x0 * old_w, pos.width * old_w
    leg_w = leg.get_window_extent().width / dpi

    new_w = old_w + leg_w + pad
    fig.set_size_inches(new_w, h)
    # set_size_inches keeps the axes at constant *fractions*, so restore inches
    ax.set_position([ax_x0 / new_w, pos.y0, ax_w / new_w, pos.height])


widen_for_legend(fig, ax, leg)
# plt.show()
plt.savefig("categories.pdf")
