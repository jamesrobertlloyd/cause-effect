# Read data files

X.train <- read.table('train.csv', header = TRUE, sep = ',')
X.valid <- read.table('valid.csv', header = TRUE, sep = ',')

# Load libraries

library(gbm)

# Go GBM!

set.seed(1234)

trees = 500

my_gbm <- gbm.fit(X.train[,2:dim(X.train)[2]], X.train[,1], n.trees=trees, distribution="gaussian", interaction.depth=5, shrinkage=0.1)
predictions <- predict.gbm(my_gbm,X.valid,trees)

# Write output

write.table(predictions, 'gbm_regression_predictions.csv', sep = ',', row.names = FALSE)

# Validation set version

set.seed(1234)

perm <- sample.int(dim(X.train)[1])
train <- perm[1:floor(0.8*dim(X.train)[1])]
test <- perm[(floor(0.8*dim(X.train)[1])+1):dim(X.train)[1]]

set.seed(1234)

trees = 10000
depth = 30
shrinkage = 0.01

my_gbm <- gbm.fit(X.train[train,2:dim(X.train)[2]], X.train[train,1], n.trees=trees, distribution="gaussian", interaction.depth=depth, shrinkage=shrinkage)
predictions <- predict.gbm(my_gbm,X.train[test,2:dim(X.train)[2]],trees)

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
