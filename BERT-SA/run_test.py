
with open("test_data/test_results.tsv", "r") as f:
    y = []
    for line in f.readlines():
        items = line.strip().split("\t")
        neg = float(items[0])
        pos = float(items[1])
        y.append(0 if neg > pos else 1)

with open("1160300607.csv", "w") as f:
    for i, l in enumerate(y):
        f.write("{},{}\n".format(i, l)) 