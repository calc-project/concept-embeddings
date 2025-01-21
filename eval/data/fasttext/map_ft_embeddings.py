import csv
import numpy as np
from pyconcepticon import Concepticon
from collections import defaultdict, Counter


# define language IDs (Concepticon -> fastText)
lang_ids = {
    "arabic": "ar",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "polish": "pl",
    "russian": "ru",
    "chinese": "zh"
}

# set up Concepticon instance and retrieve Multi-SimLex by its ID
concepticon = Concepticon()
conceptlist = list(concepticon.conceptlists["Vulic-2020-2244"].concepts.values())

# for each concept, collect the translations in the individual languages
translations = defaultdict(lambda: defaultdict(list))
vocabularies = defaultdict(set)

for concept in conceptlist:
    gloss = concept.concepticon_gloss
    if not gloss:
        continue
    for l in lang_ids:
        gloss_translations = concept.attributes.get(f"{l}_in_source", [])
        translations[gloss][l].extend(gloss_translations)
        vocabularies[l] = vocabularies[l] | set(gloss_translations)

# now look up the pretrained fasttext embeddings for each language
for lang, lang_id in lang_ids.items():
    vocab = vocabularies[lang]
    # load fasttext embeddings
    ft_embeddings = {}
    with open(f"cc.{lang_id}.300.vec") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            # only store relevant words to save memory
            if word in vocab:
                ft_embeddings[word] = [float(x) for x in values[1:]]

    table = [["CONCEPTICON_GLOSS", "TRANSLATIONS", "EMBEDDING"]]
    for concept, trans in translations.items():
        translations_in_lang = trans[lang]
        embeddings = [ft_embeddings[word] for word in translations_in_lang if word in ft_embeddings]
        if not embeddings:
            continue
        mean_embedding = np.mean(embeddings, axis=0).tolist()
        valid_translations = [x for x in translations_in_lang if x in ft_embeddings]
        c = Counter(valid_translations)
        # ensure valid json notation for word counts
        json_string = "{" + ", ".join([f'"{k}": {v}' for k, v in c.items()]) + "}"
        table.append([concept, json_string, str(mean_embedding)])

    with open(f"{lang}_embeddings.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(table)
