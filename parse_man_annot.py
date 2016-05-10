from collections import defaultdict
fname = "gateman.mpqa.lre.3.0"
f = open(fname)
annots = defaultdict(lambda: defaultdict)

for line in f:
    if line[0] != "#":
        print line
        line = line.strip().split("\t")
        ranges.append(line[1])
        annots[line[1]]["annot_type"] = line[2]
        if len(line) > 3:
            annots[line[1]]["attrs"] = line[3]
print len(ranges)
print len(annotypes)
print len(attrs)
