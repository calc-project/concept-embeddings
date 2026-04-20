from pathlib import Path

bats_dir = Path(__file__).parent / "BATS_3.0"
categories = ["E10 [male - female]", "E07 [animal - sound]", "E09 [things - color]", "E08 [animal - shelter]", "E06 [animal - young]",
              "L02 [hypernyms - misc]", "L09 [antonyms - gradable]", "L10 [antonyms - binary]", "L04 [meronyms - substance]",
              "L03 [hyponyms - misc]", "L01 [hypernyms - animals]", "L05 [meronyms - member]", "L06 [meronyms - part]"]

pairs = []
for file in (bats_dir / "3_Encyclopedic_semantics").glob("*.txt"):
    category = file.stem
    if category not in categories:
        continue
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            word, targets = line.split()
            pairs.append((word, targets, category))

for file in (bats_dir / "4_Lexicographic_semantics").glob("*.txt"):
    category = file.stem
    if category not in categories:
        continue
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            word, targets = line.split()
            pairs.append((word, targets, category))

with open(Path(__file__).parent / "analogies.tsv", "w") as f:
    f.write("BATS_WORD\tBATS_TARGETS\tBATS_CATEGORY\tCONCEPT_1\tCONCEPT_2\n")
    for pair in pairs:
        word, targets, category = pair
        f.write(f"{word}\t{targets}\t{category}\t\t\n")
