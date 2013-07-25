# Various routines to help with ensembling the predictors

import sys

def combine_features(use_training_data=False):
    #row_names = ['train%d' % i for i in range(1, 16199+1, 1)] + ['valid%d' % i for i in range(1, 4050+1, 1)]
    row_names = ['train%d' % i for i in range(1, 32398+1, 1)] + ['valid%d' % i for i in range(1, 4050+1, 1)]
    #row_names = ['train%d' % i for i in range(1, 129592+1, 1)] + ['valid%d' % i for i in range(1, 4050+1, 1)]
    # Concatenates feature files and saves is output format that is easy for random forest
    feature_files = [#'../predictors/real/benchmark_features_no_sample_size.csv',
                     #'../predictors/real/benchmark_features_ole_no_nas.csv',
                     #'../predictors/real/benchmark_features_ole_only_rank_skew_kurt.csv',
                     '../features/real/publicinfo.csv',
                     '../features/real/reasonable_features.csv',#_extended.csv',
                     '../features/real/injectivity.csv',
                     '../features/real/auto_pit_20_10.csv',
                     '../features/real/auto_pit_20_15.csv',
                     '../features/real/auto_pit_10_05.csv',
                     '../features/real/auto_pit_10_10.csv',
                     '../features/real/auto_pit_10_15.csv',
                     '../features/real/auto_pit_30_10.csv',
                     '../features/real/auto_pit_30_15.csv',
                     #'../features/real/corrs.csv',
                     #'../features/real/high_order_moments.csv',
                     #'../features/real/icgi.csv',
                     '../features/real/unreasonable_features.csv',
                     ]
                     #'../predictors/real/kendall.csv']#,
                     #'../predictors/real/moment_5.csv']]
    combined = {row_name : [] for row_name in row_names}
    feature_names = []
    for filename in feature_files:
        with open(filename, 'rU') as data:
            #data.readline() # Skip header
            feature_names += data.readline().strip().split(',')[1:]
            for line in data:
                combined[line.split(',')[0]] += line.rstrip().split(',')[1:]
    # Load all targets
    target_files = ['../data/training-flipped/CEdata_train_target.csv']#,
                    #'../data/ensemble_training/CEdata_train_ensemble_target.csv']
    combined_targets = {row_name : '' for row_name in row_names}
    for filename in target_files:
        with open(filename, 'r') as data:
            data.readline() # Skip header
            for line in data:
                if line.strip().split(',')[1] == '1':
                    combined_targets[line.split(',')[0]] = '1'
                elif line.strip().split(',')[1] == '-1':
                    combined_targets[line.split(',')[0]] = '-1'
                else:
                    combined_targets[line.split(',')[0]] = '0'
    # Save ensemble data
    #with open('../data/ensemble_training/CEdata_train_ensemble_target.csv') as data:
    with open('../data/training-flipped/CEdata_train_target.csv') as data:
        data.readline()
        ensemble_row_names = [line.split(',')[0] for line in data]
    #if use_training_data:
    #    with open('../data/training/CEdata_train_reduced_target.csv') as data:
    #        data.readline()
    #        ensemble_row_names += [line.split(',')[0] for line in data]
    lines = []
    lines.append('Target,' + ','.join(feature_names) + '\n')
    for row_name in ensemble_row_names:
        lines.append(','.join([combined_targets[row_name]] + combined[row_name]) + '\n')
    with open('../rf/train.csv', 'w') as outfile:
        outfile.writelines(lines)
    # Save validation data
    #with open('../data/kaggle_validation/CEfinal_valid_publicinfo.csv') as data:
    with open('../data/validation/CEfinal_valid_publicinfo.csv') as data:
        data.readline()
        valid_row_names = [line.split(',')[0] for line in data]
    lines = []
    lines.append(','.join(feature_names) + '\n')
    for row_name in valid_row_names:
        lines.append(','.join(combined[row_name]) + '\n')
    with open('../rf/valid.csv', 'w') as outfile:
        outfile.writelines(lines)
    
def format_rf_output():
    valid_row_names = ['valid%d' % i for i in range(1, 4050+1, 1)]
    with open('../rf/rf_predictions.csv', 'r') as data:
        data.readline() # Skip header
        lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + line for (i, line) in enumerate(data)]
    with open('../output/rf_predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
    
def format_rf_regression_output():
    valid_row_names = ['valid%d' % i for i in range(1, 4050+1, 1)]
    with open('../rf/rf_regression_predictions.csv', 'r') as data:
        data.readline() # Skip header
        lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + line for (i, line) in enumerate(data)]
    with open('../output/rf_regression_predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
    
def format_gbm_output():
    valid_row_names = ['valid%d' % i for i in range(1, 4050+1, 1)]
    with open('../rf/gbm_predictions.csv', 'r') as data:
        data.readline() # Skip header
        lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + line for (i, line) in enumerate(data)]
    with open('../output/gbm_predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
    
def format_gbm_regression_output():
    valid_row_names = ['valid%d' % i for i in range(1, 4050+1, 1)]
    with open('../rf/gbm_regression_predictions.csv', 'r') as data:
        data.readline() # Skip header
        lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + line for (i, line) in enumerate(data)]
    with open('../output/gbm_regression_predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
        
if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'combine-features':
            combine_features(True)
        elif sys.argv[1] == 'format-rf':
            format_rf_output()
    
