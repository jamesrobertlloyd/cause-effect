import cPickle as pickle

# Change this path for your filename
fin = open("data/training/CEdata_train_reduced_pairs.csv")

data = dict()

print "Build a dictionary of tuples of lists..."
for line in fin:
	tupleIndex = 0  # Start but putting into a	
	firstComma = line.index(',')
	secondComma = line.index(',',firstComma + 1)
	testNum = line[0:firstComma]

	data[testNum] = ([],[]) # Create empty dict of tuples of lists 

	line = line[firstComma+1:].split()
	for item in line:
		data[testNum][tupleIndex].append(item)	
		if item[-1] == ',':
# 			data[testNum][tupleIndex].append(item)
			tupleIndex = 1  # Switch to b
print "Dictionary produced with", len(data), "items"

# I'd love to pickle this out, but I don't have the RAM for it (16GB!)

# Pattern ideas...
# 1. My vertical bucket idea

for row in data:
	print min(row[0]), max(row[0]), min(row[1]), max(row[1])