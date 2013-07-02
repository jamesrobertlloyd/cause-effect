import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference)),
                ('Entropy Ratio', ['A','B'], f.MultiColumnTransform(f.entropy_ratio)),
                ('Spearman rank correlation', ['A','B'], f.MultiColumnTransform(f.rcorrelation)),
                ('Kurtosis A', 'A', f.SimpleTransform(transformer=f.fkurtosis)),
                ('Kurtosis B', 'B', f.SimpleTransform(transformer=f.fkurtosis)),
                ('Unique ratio A', 'A', f.SimpleTransform(transformer=f.unique_ratio)),
                ('Unique ratio B', 'B', f.SimpleTransform(transformer=f.unique_ratio)),
                ('Skew A', 'A', f.SimpleTransform(transformer=f.fskew)),
                ('Skew B', 'B', f.SimpleTransform(transformer=f.fskew))]
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
    feature_names = ['A: Normalized Entropy',
                     'B: Normalized Entropy',
                     'Pearson R',
                     'Pearson R Magnitude',
                     'Entropy Difference',
                     'Entropy Ratio',
                     'Spearman rank correlation',
                     'Kurtosis A',
                     'Kurtosis B',
                     'Unique ratio A',
                     'Unique ratio B',
                     'Skew A',
                     'Skew B']
    data_io.write_real_features('reasonable_features', all_names, all_features, feature_names)
    
if __name__=="__main__":
    main()
