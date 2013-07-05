# Read data files

X.train <- read.table('train.csv', header = TRUE, sep = ',')
X.valid <- read.table('valid.csv', header = TRUE, sep = ',')

X.train <- X.train[1:5000,]

# Random forest it

library(randomForest)

# Go random forest!

#### TODO - try bagging with 95% data if increase in data helps - OOB error decreases significantly

set.seed(1234)
rf.AB <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]==1), xtest = X.valid, replace = TRUE, do.trace = 100, ntree = 5000, importance=TRUE)
predictions.AB <- rf.AB$test$votes[,2]
sort(rf.AB$importance[,4])

#set.seed(1234)
#rf.BA <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]==-1), xtest = X.valid, replace = TRUE, do.trace = 100, ntree = 5000, importance=TRUE)
#predictions.BA <- rf.BA$test$votes[,2]

#set.seed(1234)
#rf <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]), xtest = X.valid, replace = TRUE, do.trace = 1000, ntree = 10000, importance=TRUE)
#predictions <- rf$test$votes[,3] - rf$test$votes[,1]

# Write output

write.table(predictions.AB, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(1-predictions.BA, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(predictions.AB-predictions.BA, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(exp(predictions.AB)-exp(predictions.BA), 'rf_predictions.csv', sep = ',', row.names = FALSE)

# Write output

#write.table(predictions, 'rf_predictions.csv', sep = ',', row.names = FALSE)

