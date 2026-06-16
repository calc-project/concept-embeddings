import json
from pathlib import Path

from graphembeddings.utils.postprocess import fuse_embeddings
from graphembeddings.utils.io import read_embeddings

from graphembeddings.eval.multisimlex import msl_correlation, read_msl_data
from graphembeddings.eval.semshift import load_shifts, sample_random_shifts, generate_training_data, fit_logistic_regression

# set up MSL
msl = read_msl_data("eval/data/msl/multisimlex.csv")

BASE_DIR = Path(__file__).parent / "output"

combinations = ["clics4-clips"]
# models = ["n2v-cbow", "n2v-sg", "prone", "sdne"]
# models = ["semantic-node2vec-sbert", "semantic-node2vec-2", "prone"]
models = ["node2vec_sg_ns", "node2vec_sg_ns_20", "node2vec_sg_nons"]

for c in combinations:
    # create target directory
    #target_dir = BASE_DIR / c
    #try:
    #    target_dir.mkdir()
    #except FileExistsError:
    #    pass

    for model in models:
        # data = [read_embeddings(BASE_DIR / f"{method}fams" / f"{model}.json", metadata=True) for method in c.split("-")]
        # data = [read_embeddings(BASE_DIR / f"{model}-{method}-mixed-2.json", metadata=True) for method in c.split("-")]
        data = [read_embeddings(BASE_DIR / method / f"{model}.json", metadata=True) for method in c.split("-")]
        embeddings = [x["embeddings"] for x in data]
        metadata = [x["parameters"] for x in data]
        fused_embeddings = fuse_embeddings(*embeddings)
        # convert np arrays back to regular lists, so they can be serialized
        fused_embeddings = {k: v.tolist() for k, v in fused_embeddings.items()}
        output_data = {"parameters": metadata, "embeddings": fused_embeddings}

        concepts = list(fused_embeddings.keys())
        out_dir = Path(BASE_DIR / c)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{model}.json", "w") as f:
            json.dump(output_data, f)

        # set up SemShift
        shifts = load_shifts("eval/data/semshift/shift_summary.tsv", embeddings=concepts)
        random_shifts = []
        for _ in range(50):
            random_shifts.append(sample_random_shifts(shifts, concepts)[1])

        print(model, c)
        print("MSL Correlation:", msl_correlation(msl, fused_embeddings))
        # eval SemShift
        total_acc = 0
        for random in random_shifts:
            X, y = generate_training_data(shifts, random, fused_embeddings)
            lr = fit_logistic_regression(X, y)
            total_acc += lr.score(X, y)
        print(f"SemShift mean acc.: {total_acc / len(random_shifts):.4f}")
