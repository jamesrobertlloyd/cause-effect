# Read data files

X.train <- read.table('train_auto.csv', header = TRUE, sep = ',')

# Random forest it

library(randomForest)

# Go random forest!

set.seed(1234)

trees = 5000

rf.AB <- randomForest(X.train[,seq(2,dim(X.train)[2])], as.factor(X.train[,1]==1), replace = TRUE, do.trace = 50, ntree = trees, importance=TRUE, keep.forest=FALSE)

sort(rf.AB$importance[,4])

