from sklearn.decomposition import PCA


langs = ["ar", "en", "es", "et", "fi", "fr", "pl", "ru", "zh"]


for l in langs:
    # read in embeddings
    words, embeddings = [], []
    with open(f"cc.{l}.300.vec") as f:
        for line in f:
            line = line.split()
            if len(line) != 301:
                continue
            words.append(line[0])
            emb = [float(x) for x in line[1:]]
            embeddings.append(emb)

    # perform PCA
    pca = PCA(n_components=128)
    reduced_embeddings = pca.fit_transform(embeddings)

    # write reduced embeddings to file
    with open(f"cc.{l}.128.vec", "w") as f:
        for word, emb in zip(words, reduced_embeddings):
            f.write(f"{word} {' '.join(map(str, emb))}\n")
