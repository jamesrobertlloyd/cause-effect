# Read data files

X.train <- read.table('train.csv', header = FALSE, sep = ',')
X.valid <- read.table('valid.csv', header = FALSE, sep = ',')

# Random forest it

library(randomForest)

# Go random forest!

rf <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]), xtest = X.valid, replace = TRUE, do.trace = 100, ntree = 1000, importance=TRUE)
predictions <- rf$test$votes[,2]

# Write output

write.table(predictions, 'rf_predictions.csv', sep = ',', row.names = FALSE)

