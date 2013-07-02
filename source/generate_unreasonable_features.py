import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('Max A', 'A', f.SimpleTransform(transformer=max)),
                ('Max B', 'B', f.SimpleTransform(transformer=max)),
                ('Min A', 'A', f.SimpleTransform(transformer=min)),
                ('Min B', 'B', f.SimpleTransform(transformer=min)),
                ('Mean A', 'A', f.SimpleTransform(transformer=f.mean)),
                ('Mean B', 'B', f.SimpleTransform(transformer=f.mean)),
                ('Sd A', 'A', f.SimpleTransform(transformer=f.sd)),
                ('Sd B', 'B', f.SimpleTransform(transformer=f.sd))]
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
    feature_names = ['Number of samples',
                     'Max A',
                     'Max B',
                     'Min A',
                     'Min B',
                     'Mean A',
                     'Mean B',
                     'Sd A',
                     'Sd B']
    data_io.write_real_features('unreasonable_features', all_names, all_features, feature_names)
    
if __name__=="__main__":
    main()
