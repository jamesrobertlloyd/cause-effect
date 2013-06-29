# Various routines to help with ensembling the predictors

def combine_features:
    row_names = ['train%d' % i for i in range(1, 7830+1, 1)] + ['valid%d' % i for i in range(1, 2642+1, 1)]
    # Concatenates feature files and saves is output format that is easy for random forest
    feature_files = ['../predictors/real/benchmark_features.csv']
    
