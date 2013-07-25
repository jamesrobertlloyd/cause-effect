# Read data files

X.train <- read.table('train.csv', header = TRUE, sep = ',')
X.valid <- read.table('valid.csv', header = TRUE, sep = ',')

# Random forest it

library(randomForest)

# Go random forest!

set.seed(1234)

set.seed(1234)
rf <- randomForest(X.train[,2:dim(X.train)[2]], X.train[,1], xtest = X.valid, replace = TRUE, do.trace = 10, ntree = 5000, importance=TRUE, keep.forest=FALSE)
predictions <- rf$test$predicted

sort(rf$importance[,2])

# Write output

write.table(predictions, 'rf_regression_predictions.csv', sep = ',', row.names = FALSE)

# Validation version

set.seed(1234)

perm <- sample.int(dim(X.train)[1])
train <- perm[1:floor(0.8*dim(X.train)[1])]
test <- perm[(floor(0.8*dim(X.train)[1])+1):dim(X.train)[1]]

set.seed(1234)

trees = 10000
mtry = 65
sampsize = 17500

rf <- randomForest(X.train[train,2:dim(X.train)[2]], X.train[train,1], xtest = X.train[test,2:dim(X.train)[2]], ytest=X.train[test,1], replace = TRUE, do.trace = 5, ntree = trees, importance=TRUE, keep.forest=FALSE, mtry=mtry, sampsize=sampsize)
predictions <- rf$test$predicted

truth <- X.train[test,1]

auc <- function(outcome, proba){
  N = length(proba)
  N_pos = sum(outcome)
  df = data.frame(out = outcome, prob = proba)
  df = df[order(-df$prob),]
  df$above = (1:N) - cumsum(df$out)
  return( 1- sum( df$above * df$out ) / (N_pos * (N-N_pos) ) )
}

bi.auc <- function(outcome, proba){
  return (0.5 * (auc(1*(outcome==1), proba) + auc(1*(outcome==-1), -proba)))
}

bi.auc(truth, predictions)

