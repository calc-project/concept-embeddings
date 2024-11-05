import csv
from pyconcepticon import Concepticon
from tabulate import tabulate


cnc = Concepticon() # add path to concepticon if you did not run `cldfbench catconfig`
# get the concept list by
cl = {
        concept.number: concept for concept in cnc.conceptlists[
                "Vulic-2020-2244"].concepts.values()}

# this little dictionary will handle keys to all data in Multi-Simlex
msl = {}
for concept in cl.values():
    for (idx, link, eng, rus, chin, can, ara, span, pol, fra, est, fin) in zip(
            concept.attributes["simlex_ids"],
            concept.attributes["links"],
            concept.attributes["english_score"],
            concept.attributes["russian_score"],
            concept.attributes["chinese_score"],
            concept.attributes["cantonese_score"],
            concept.attributes["arabic_score"],
            concept.attributes["spanish_score"],
            concept.attributes["polish_score"],
            concept.attributes["french_score"],
            concept.attributes["estonian_score"],
            concept.attributes["finnish_score"]
        ):
        msl[idx] = [
                concept.concepticon_id or "",
                concept.concepticon_gloss or "",
                eng, rus, chin, can, ara, span, pol, fra, est, fin
                ]

with open("multisimlex.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["CID1", "CGL1", "CID2", "CGL2", "eng", "rus", "cmn", "yue", "ara",
                     "spa", "pol", "fra", "est", "fin", "mean"])
    for i in range(len(msl) // 2):
        row1 = msl[f"{i+1}:1"]
        row2 = msl[f"{i+1}:2"]
        if not (row1[0] and row2[0]):
            continue
        mean = sum(row1[2:]) / len(row1[2:])
        combined = row1[:2] + row2 + [mean]
        writer.writerow(combined)
