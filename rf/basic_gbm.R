# Read data files

X.train <- read.table('train.csv', header = TRUE, sep = ',')
X.valid <- read.table('valid.csv', header = TRUE, sep = ',')

# Load libraries

library(gbm)

# Go GBM!

set.seed(1234)

trees = 10000

my_gbm <- gbm.fit(X.train[,2:dim(X.train)[2]], 1*(X.train[,1]==1), n.trees=trees, distribution="bernoulli", interaction.depth=3, shrinkage=0.01)
p1 <- predict.gbm(my_gbm,X.valid,trees)

my_gbm <- gbm.fit(X.train[,2:dim(X.train)[2]], 1*(X.train[,1]==-1), n.trees=trees, distribution="bernoulli", interaction.depth=3, shrinkage=0.01)
p3 <- predict.gbm(my_gbm,X.valid,trees)

predictions <- p1 - p3

# Write output

write.table(predictions, 'gbm_predictions.csv', sep = ',', row.names = FALSE)

