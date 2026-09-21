import collections
import pycldf
import csv


# load types of semantic change
type_per_shift = {}
with open("TypesofSemanticChangeData.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        shift_id = row["ID"]
        shift_type = row["Frame"] or row["Extension"] or row["Emotive Value"]
        type_per_shift[shift_id] = shift_type

shiftset = collections.defaultdict(list)

ds = pycldf.Dataset.from_metadata("datsemshift/cldf/cldf-metadata.json")

# load the shifts from the lexeme table

concepts = ds.objects("ParameterTable")
gloss2con = {
        c.data["Name"]: c.data["Concepticon_Gloss"] for c in concepts if c}

shifts = []
with open("datsemshift/raw/lexemes.tsv") as f:
    for row in f:
        row = [c.strip() for c in row.split("\t")]
        if row[10] != row[12] or row[0] == "ID":
            shifts += [row]
selected_shifts = [[
    "ID",
    "Shift_ID",
    "Source_Language",
    "Target_Language",
    "Source_Gloss",
    "Target_Gloss",
    "Source_Meaning",
    "Target_Meaning",
    "Source_Form",
    "Target_Form",
    "Shift_Type"
    ]]
count = 0
idx = 1
for row_ in shifts[1:]:
    row = collections.OrderedDict(zip(shifts[0], row_))
    c1, c2 = gloss2con.get(row["Source_Concept"]), gloss2con.get(row["Target_Concept"])
    if c1 and c2:
        selected_shifts += [
            [
                str(idx),
                row["ID"],
                row["Source_Language"],
                row["Target_Language"],
                c1,
                c2,
                row["Source_Meaning"],
                row["Target_Meaning"],
                row["Source_Word"],
                row["Target_Word"],
                type_per_shift.get(row["Shift_ID"], ""),
            ]
        ]

        count += 1
        idx += 1

# write all shifts to file
with open("shifts_with_examples.tsv", "w") as f:
    for row in selected_shifts:
        f.write("\t".join(row) + "\n")

# create and write summary of shifts
combined = collections.defaultdict(list)
for row_ in selected_shifts[1:]:
    row = collections.OrderedDict(zip(selected_shifts[0], row_))
    combined[row["Source_Gloss"], row["Target_Gloss"]] += [row]
with open("shift_summary.tsv", "w") as f:
    f.write("Source\tTarget\tCount\tShift_Type\tIDS\n")
    for (a, b), row in sorted(combined.items(), key=lambda x: len(x[1]),
                              reverse=True):
        f.write(f"{a}\t{b}\t{len(row)}\t{row[0]["Shift_Type"]}\t{" ".join([r["ID"] for r in row])}\n")
        #f.write("{0}\t{1}\t{2}\t{3}\n".format(
        #    a, b, len(row), " ".join([r["ID"] for r in row])))