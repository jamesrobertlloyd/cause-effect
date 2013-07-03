import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('Kendall tau', ['A','B'], f.MultiColumnTransform(f.kendall)),
                ('Kendall tau p', ['A','B'], f.MultiColumnTransform(f.kendall_p)),
                ('Mann Whitney', ['A','B'], f.MultiColumnTransform(f.mannwhitney)),
                ('Mann Whitney p', ['A','B'], f.MultiColumnTransform(f.mannwhitney_p)),
                ('Wilcoxon', ['A','B'], f.MultiColumnTransform(f.wilcoxon)),
                ('Wilcoxon p', ['A','B'], f.MultiColumnTransform(f.wilcoxon_p)),
                ('Kruskal', ['A','B'], f.MultiColumnTransform(f.kruskal)),
                ('Kruskal p', ['A','B'], f.MultiColumnTransform(f.kruskal_p))]
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
    feature_names = ['Kendall tau',
                     'Kendall tau p',
                     'Mann Whitney',
                     'Mann Whitney p',
                     'Wilcoxon',
                     'Wilcoxon p',
                     'Kruskal',
                     'Kruskal p']
    data_io.write_real_features('corrs', all_names, all_features, feature_names)
    
if __name__=="__main__":
    main()
