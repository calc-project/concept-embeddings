from pathlib import Path
from graphembeddings.eval.multisimlex import msl_correlation, read_msl_data
from graphembeddings.eval.semshift import load_shifts, sample_random_shifts, generate_training_data, fit_logistic_regression
from graphembeddings.eval.eat import load_eat_edges


DEFAULT_EVAL_DATA_DIR = Path(__file__).parent.parent.parent.parent / "eval" / "data"

class Evaluation:
    def __init__(self, concepts, eval_data_dir=None):
        self.concepts = concepts
        self.eval_data_dir = eval_data_dir or DEFAULT_EVAL_DATA_DIR
        # set up MultiSimLex data
        self.msl = read_msl_data(self.eval_data_dir / "msl" / "multisimlex.csv")
        # set up SemShift
        self.shifts = load_shifts(self.eval_data_dir / "semshift" / "shift_summary.tsv", embeddings=self.concepts)
        self.random_shifts = [sample_random_shifts(self.shifts, self.concepts)[1] for _ in range(50)]
        # set up EAT
        self.eat_edges, _ = load_eat_edges(self.eval_data_dir / "norare" / "norare-data")
        self.random_edges = [sample_random_shifts(self.eat_edges, self.concepts)[1] for _ in range(50)]

    def eval_msl(self, embeddings):
        return msl_correlation(self.msl, embeddings)

    def eval_semshift(self, embeddings):
        total_acc = 0
        for random in self.random_shifts:
            X, y = generate_training_data(self.shifts, random, embeddings)
            lr = fit_logistic_regression(X, y)
            total_acc += lr.score(X, y)

        return total_acc / len(self.random_shifts)

    def eval_eat(self, embeddings):
        total_acc = 0
        for random in self.random_edges:
            X, y = generate_training_data(self.eat_edges, random, embeddings)
            lr = fit_logistic_regression(X, y)
            total_acc += lr.score(X, y)

        return total_acc / len(self.random_edges)

    def eval_all(self, embeddings):
        return self.eval_msl(embeddings), self.eval_semshift(embeddings), self.eval_eat(embeddings)
