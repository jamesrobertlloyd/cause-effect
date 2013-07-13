# Read data files

X.train <- read.table('train.csv', header = TRUE, sep = ',')
X.valid <- read.table('valid.csv', header = TRUE, sep = ',')

# Random forest it

library(randomForest)

# Go random forest!

set.seed(1234)

high.memory = FALSE

#feature.subset <- c(6,1,2,10,50,11,17,14,44,46,20,51,12,45,48,29,33,8,43,13)
#feature.subset <- c(6,1,2,10,50,11,17,14,44,46,20,51,12,45,48,29,33,8,43,13,24,25,26,27,28,30,47,51,52,53)
#feature.subset <- c(6,1,2,10,50,11,17,14,44,46,20,51,12,45,48,29,33,8,43,13,24,25,26,27,28,30,47,51,52,53,9,5,22,21,23)
#feature.subset <- c(seq(1,30), seq(43,51), 33)
feature.subset <- seq(1,dim(X.valid)[2])
trees = 100000

# Rejected first time
# 9, 5, 22, 21, 23, 7
# Marginal accept
# 12, 43, 13
# Marginal reject
# 48, 3, 4, 15, 16, 18, 19, 20

#if (FALSE)#(file.exists('saved-forest-AB.RData'))
#{
#    load('saved-forest-AB.RData')
#    predictions.AB <- predict(rf.AB, X.valid, type='prob')[,2]
#} else
#{
#    if (high.memory) {
#        #rf.AB <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]==1), xtest = X.valid, replace = TRUE, do.trace = 50, ntree = trees, importance=TRUE, keep.forest=TRUE)
#        rf.AB <- randomForest(X.train[,1+feature.subset], as.factor(X.train[,1]==1), xtest = X.valid[,feature.subset], replace = TRUE, do.trace = 50, ntree = trees, importance=TRUE, keep.forest=TRUE)
#        save(rf.AB, 'rf.AB', file = 'saved-forest-AB.RData')
#        predictions.AB <- rf.AB$test$votes[,2]
#    } else
#    {
        #rf.AB <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]==1), xtest = X.valid, replace = TRUE, do.trace = 50, ntree = trees, importance=TRUE, keep.forest=FALSE)
#        rf.AB <- randomForest(X.train[,1+feature.subset], as.factor(X.train[,1]==1), xtest = X.valid[,feature.subset], replace = TRUE, do.trace = 50, ntree = trees, importance=TRUE, keep.forest=FALSE)
#        predictions.AB <- rf.AB$test$votes[,2]
#    }
#}

#sort(rf.AB$importance[,4])

#set.seed(1234)
#rf.BA <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]==-1), xtest = X.valid, replace = TRUE, do.trace = 100, ntree = 5000, importance=TRUE, keep.forest=TRUE)
#predictions.BA <- rf.BA$test$votes[,2]

set.seed(1234)
rf <- randomForest(X.train[,2:dim(X.train)[2]], as.factor(X.train[,1]), xtest = X.valid, replace = TRUE, do.trace = 50, ntree = 20000, importance=TRUE, keep.forest=FALSE)
predictions <- rf$test$votes[,3] - rf$test$votes[,1]

# Write output

#write.table(predictions.AB, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(1-predictions.BA, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(predictions.AB-predictions.BA, 'rf_predictions.csv', sep = ',', row.names = FALSE)
#write.table(exp(predictions.AB)-exp(predictions.BA), 'rf_predictions.csv', sep = ',', row.names = FALSE)

# Write output

write.table(predictions, 'rf_predictions.csv', sep = ',', row.names = FALSE)

