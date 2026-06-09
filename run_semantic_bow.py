from graphembeddings.models.trainer import Node2Vec
from graphembeddings.utils.preprocess import BOWEncoder
from graphembeddings.utils.io import read_graph_data
from pathlib import Path
from graphembeddings.eval.multisimlex import msl_correlation, read_msl_data
from graphembeddings.eval.semshift import load_shifts, sample_random_shifts, generate_training_data, fit_logistic_regression


DATA_DIR = Path(__file__).parent / "data" / "graphs"
OUTPUT_DIR = Path(__file__).parent / "output"

# set up MSL
msl = read_msl_data("eval/data/msl/multisimlex.csv")

for mode in ["full", "affix", "overlap"]:
    DATA_FP = DATA_DIR / f"{mode}fams.json"
    _, _, concept_to_id = read_graph_data(DATA_FP)
    concepts = list(concept_to_id.keys())

    # set up SemShift
    shifts = load_shifts("eval/data/semshift/shift_summary.tsv", embeddings=concepts)
    random_shifts = []
    for _ in range(50):
        random_shifts.append(sample_random_shifts(shifts, concepts)[1])

    # Node2Vec (SkipGram) -- semantic BOW encoding
    node2vec = Node2Vec.from_graph_file(DATA_FP)
    encoder = BOWEncoder(node2vec.id_to_concept.values(), min_token_count=2, keep_one_hot=True)
    concept_to_id = {c: i for i, c in node2vec.id_to_concept.items()}
    encodings = encoder.generate_encoding_matrix(concept_to_id)
    print(f"BOW Size: {len(encodings[0])} with n={1}")
    node2vec.train(cbow=False, max_epochs=1500, patience=5, min_delta=0.001, encodings=encodings)
    node2vec.save(OUTPUT_DIR / f"semantic-{mode}-mixed-2.json")

    embeddings = node2vec.embeddings
    # eval MSL
    print("MSL Correlation:", msl_correlation(msl, embeddings))
    # eval SemShift
    total_acc = 0
    for random in random_shifts:
        X, y = generate_training_data(shifts, random, embeddings)
        lr = fit_logistic_regression(X, y)
        total_acc += lr.score(X, y)
    print(f"SemShift mean acc.: {total_acc / len(random_shifts):.4f}")

    # print("Done training Node2Vec (Semantic Encoding).")

# Node2Vec (SkipGram) -- one-hot encoding
# node2vec = Node2Vec.from_graph_file(DATA_FP)
# node2vec.train(cbow=False, max_epochs=1500, patience=5, min_delta=0.001)
# node2vec.save(OUTPUT_DIR / "one-hot.json")
# print("Done training Node2Vec (One-Hot Encoding).")

