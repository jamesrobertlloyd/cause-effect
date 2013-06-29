# Various routines to help with ensembling the predictors

def combine_features():
    #row_names = ['train%d' % i for i in range(1, 19809+1, 1)] + ['valid%d' % i for i in range(1, 6075+1, 1)]
    row_names = ['train%d' % i for i in range(1, 19809+1, 1)] + ['valid%d' % i for i in range(1, 2642+1, 1)]
    # Concatenates feature files and saves is output format that is easy for random forest
    feature_files = ['../predictors/real/benchmark_features.csv']
    combined = {row_name : [] for row_name in row_names}
    for filename in feature_files:
        with open(filename, 'r') as data:
            data.readline() # Skip header
            for line in data:
                combined[line.split(',')[0]] += line.rstrip().split(',')[1:]
    # Load all targets
    target_files = ['../data/training/CEdata_train_reduced_target.csv',
                    '../data/ensemble_training/CEdata_train_ensemble_target.csv']
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
    with open('../data/ensemble_training/CEdata_train_ensemble_target.csv') as data:
        data.readline()
        ensemble_row_names = [line.split(',')[0] for line in data]
    lines = []
    for row_name in ensemble_row_names:
        lines.append(','.join([combined_targets[row_name]] + combined[row_name]) + '\n')
    with open('../rf/train.csv', 'w') as outfile:
        outfile.writelines(lines)
    # Save validation data
    with open('../data/kaggle_validation/CEdata_valid_publicinfo.csv') as data:
        data.readline()
        valid_row_names = [line.split(',')[0] for line in data]
    lines = []
    for row_name in valid_row_names:
        lines.append(','.join(combined[row_name]) + '\n')
    with open('../rf/valid.csv', 'w') as outfile:
        outfile.writelines(lines)
    
def format_rf_output():
    #valid_row_names = ['valid%d' % i for i in range(1, 6075+1, 1)]
    valid_row_names = ['valid%d' % i for i in range(1, 2642+1, 1)]
    with open('../rf/rf_predictions.csv', 'r') as data:
        data.readline() # Skip header
        lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + line for (i, line) in enumerate(data)]
    with open('../output/predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
    
