import json
from pathlib import Path

from graphembeddings.utils.postprocess import fuse_embeddings
from graphembeddings.utils.io import read_embeddings


BASE_DIR = Path(__file__).parent / "embeddings"

combinations = ["full-affix", "full-overlap", "full-affix-overlap"]
models = ["n2v-cbow", "n2v-sg", "prone", "sdne"]

for c in combinations:
    # create target directory
    target_dir = BASE_DIR / c
    try:
        target_dir.mkdir()
    except FileExistsError:
        pass

    for model in models:
        data = [read_embeddings(BASE_DIR / f"{method}fams" / f"{model}.json", metadata=True) for method in c.split("-")]
        embeddings = [x["embeddings"] for x in data]
        metadata = [x["parameters"] for x in data]
        fused_embeddings = fuse_embeddings(*embeddings)
        # convert np arrays back to regular lists, so they can be serialized
        fused_embeddings = {k: v.tolist() for k, v in fused_embeddings.items()}
        output_data = {"parameters": metadata, "embeddings": fused_embeddings}
        with open(target_dir / f"{model}.json", "w") as f:
            json.dump(output_data, f)
