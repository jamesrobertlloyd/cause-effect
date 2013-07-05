import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('Moment 5 A', 'A', f.SimpleTransform(transformer=f.standard_moment_5)),
                ('Moment 5 B', 'B', f.SimpleTransform(transformer=f.standard_moment_5)),
                ('Moment 5 diff', ['A','B'], f.MultiColumnTransform(f.standard_moment_diff_5)),
                ('Moment 5 ratio', ['A','B'], f.MultiColumnTransform(f.standard_moment_ratio_5)),
                ('Moment 6 A', 'A', f.SimpleTransform(transformer=f.standard_moment_6)),
                ('Moment 6 B', 'B', f.SimpleTransform(transformer=f.standard_moment_6)),
                ('Moment 6 diff', ['A','B'], f.MultiColumnTransform(f.standard_moment_diff_6)),
                ('Moment 6 ratio', ['A','B'], f.MultiColumnTransform(f.standard_moment_ratio_6)),
                ('Moment 7 A', 'A', f.SimpleTransform(transformer=f.standard_moment_7)),
                ('Moment 7 B', 'B', f.SimpleTransform(transformer=f.standard_moment_7)),
                ('Moment 7 diff', ['A','B'], f.MultiColumnTransform(f.standard_moment_diff_7)),
                ('Moment 7 ratio', ['A','B'], f.MultiColumnTransform(f.standard_moment_ratio_7)),
                ('Moment 8 A', 'A', f.SimpleTransform(transformer=f.standard_moment_8)),
                ('Moment 8 B', 'B', f.SimpleTransform(transformer=f.standard_moment_8)),
                ('Moment 8 diff', ['A','B'], f.MultiColumnTransform(f.standard_moment_diff_8)),
                ('Moment 8 ratio', ['A','B'], f.MultiColumnTransform(f.standard_moment_ratio_8)),
                ('Moment 9 A', 'A', f.SimpleTransform(transformer=f.standard_moment_9)),
                ('Moment 9 B', 'B', f.SimpleTransform(transformer=f.standard_moment_9)),
                ('Moment 9 diff', ['A','B'], f.MultiColumnTransform(f.standard_moment_diff_9)),
                ('Moment 9 ratio', ['A','B'], f.MultiColumnTransform(f.standard_moment_ratio_9))]
    combined = f.FeatureMapper(features)
    return combined

def main():
    extractor = feature_extractor()
    
    print("Reading in the training data")
    train = data_io.read_train_pairs()

    print("Extracting features from training data")
    train_features = extractor.fit_transform(train[:])
    
    print("Reading in the ensemble training data")
    ensemble_train = data_io.read_ensemble_train_pairs()

    print("Extracting features from ensemble training data")
    ensemble_train_features = extractor.fit_transform(ensemble_train[:])
    
    print("Reading in the validation data")
    valid = data_io.read_valid_pairs()

    print("Extracting features from validation data")
    valid_features = extractor.fit_transform(valid[:])
    
    all_features = np.concatenate((train_features, ensemble_train_features, valid_features))
    
    print("Concatenating names")
    train_names = [train.irow(i).name for i in range(len(train))]
    ensemble_train_names = [ensemble_train.irow(i).name for i in range(len(ensemble_train))]
    valid_names = [valid.irow(i).name for i in range(len(valid))]
    all_names = train_names + ensemble_train_names + valid_names
    
    print("Writing feature file")
    feature_names = ['Moment 5 A',
                     'Moment 5 B',
                     'Moment 5 diff',
                     'Moment 5 ratio',
                     'Moment 6 A',
                     'Moment 6 B',
                     'Moment 6 diff',
                     'Moment 6 ratio',
                     'Moment 7 A',
                     'Moment 7 B',
                     'Moment 7 diff',
                     'Moment 7 ratio',
                     'Moment 8 A',
                     'Moment 8 B',
                     'Moment 8 diff',
                     'Moment 8 ratio',
                     'Moment 9 A',
                     'Moment 9 B',
                     'Moment 9 diff',
                     'Moment 9 ratio',]
    data_io.write_real_features('high_order_moments', all_names, all_features, feature_names)
    
if __name__=="__main__":
    main()
