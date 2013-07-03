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
                ('Spearman rank magnitude', ['A','B'], f.MultiColumnTransform(f.rcorrelation_magnitude)),
                ('Kurtosis A', 'A', f.SimpleTransform(transformer=f.fkurtosis)),
                ('Kurtosis B', 'B', f.SimpleTransform(transformer=f.fkurtosis)),
                ('Kurtosis difference', ['A', 'B'], f.MultiColumnTransform(transformer=f.fkurtosis_diff)),
                ('Kurtosis ratio', ['A', 'B'], f.MultiColumnTransform(transformer=f.fkurtosis_ratio)),
                ('Unique ratio A', 'A', f.SimpleTransform(transformer=f.unique_ratio)),
                ('Unique ratio B', 'B', f.SimpleTransform(transformer=f.unique_ratio)),
                ('Skew A', 'A', f.SimpleTransform(transformer=f.fskew)),
                ('Skew B', 'B', f.SimpleTransform(transformer=f.fskew)),
                ('Skew difference', ['A', 'B'], f.MultiColumnTransform(transformer=f.fskew_diff)),
                ('Skew ratio', ['A', 'B'], f.MultiColumnTransform(transformer=f.fskew_ratio)),
                ('Pearson - Spearman', ['A', 'B'], f.MultiColumnTransform(transformer=f.Pearson_Spearman_diff)),
                ('Abs Pearson - Spearman', ['A', 'B'], f.MultiColumnTransform(transformer=f.Pearson_Spearman_abs_diff)),
                ('Pearson / Spearman', ['A', 'B'], f.MultiColumnTransform(transformer=f.Pearson_Spearman_ratio))]
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
                     'Spearman rank magnitude',
                     'Kurtosis A',
                     'Kurtosis B',
                     'Kurtosis difference',
                     'Kurtosis ratio',
                     'Unique ratio A',
                     'Unique ratio B',
                     'Skew A',
                     'Skew B',
                     'Skew difference',
                     'Skew ratio',
                     'Pearson - Spearman',
                     'Abs Pearson - Spearman',
                     'Pearson / Spearman']
    data_io.write_real_features('reasonable_features', all_names, all_features, feature_names)
    
if __name__=="__main__":
    main()
