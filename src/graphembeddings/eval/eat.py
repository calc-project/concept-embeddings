from pathlib import Path
from pynorare import NoRaRe


GRAPH_EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "embeddings"
FT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "fasttext"

NORARE_DEFAULT_FP = Path(__file__).parent.parent / "data" / "norare" / "norare-data"


def load_eat_edges(fp=NORARE_DEFAULT_FP, threshold=5):
    norare = NoRaRe(fp)
    eat = norare.datasets.get("Kiss-1973-EAT")

    edges, weights = [], []
    visited = set()
    overflow = ""

    for row in eat.concepts.values():
        c1 = row["concepticon_gloss"]
        for edge in row.get("edges", []):
            # this is necessary for handling the apostrophe, which is represented by its hex code
            if not ":" in edge:
                overflow = edge.replace("&#39", "'")
                continue
            c2, weight = edge.split(":")
            if overflow:
                c2 = overflow + c2
            overflow = ""
            weight = int(weight)
            if weight > threshold and c2 not in visited:
                edges.append((c1, c2))
                weights.append(weight)
        visited.add(c1)

    return edges, weights
