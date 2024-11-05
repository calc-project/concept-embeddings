from pycldf import Dataset
from collections import defaultdict


acd = Dataset.from_metadata('https://raw.githubusercontent.com/lexibank/acd/refs/heads/main/cldf/cldf-metadata.json')

cogsets = defaultdict(list)

for cog in acd.objects("CognateTable"):
    cog_id = cog.cognateset.id
    concept = cog.form.parameter.data.get("Concepticon_Gloss")
    if concept:
        cogsets[cog_id].append(concept)

with open("acd_cogsets.tsv", "w") as f:
    for cog_id, concepts in cogsets.items():
        f.write(f"{cog_id}\t{concepts}\n")
