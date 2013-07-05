from string import *

with open('data/ensemble_training/CEdata_train_ensemble_pairs.csv', 'r') as pairs_data_file:
	header = pairs_data_file.readline()
	body = pairs_data_file.readlines()

rows = len(body)

globalMax = 0.0
globalMin = 0.0
maxLen = 0

row = 1
while row < rows:
	splitRow = split(body[row])[1::]
	localMax = max(splitRow)
	localMin = min(splitRow)
	
	if len(splitRow) > maxLen:
		print len(splitRow)
		maxLen = len(splitRow)
	
# 	
# 	if localMin < globalMin:
# 		globalMin = localMin
# 		print globalMin
# 	
# 	if localMax > globalMax:
# 		globalMax = localMax
# 		print globalMax

	row += 1

print localMin, globalMin
# col = 1
# while col <= 5:
# 	print split(body[row], ' ')[col]
# 	col += 1