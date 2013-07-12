import numpy as np
import os.path

import data_io
import features as f

def main(overwrite=False):

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'corrs.csv')):
        print 'Feature file already exists - not overwriting'
        return

    features = [('Kendall tau', ['A','B'], f.kendall),
                ('Kendall tau p', ['A','B'], f.kendall_p),
                ('Mann Whitney', ['A','B'], f.mannwhitney),
                ('Mann Whitney p', ['A','B'], f.mannwhitney_p),
                #('Wilcoxon', ['A','B'], f.wilcoxon),
                #('Wilcoxon p', ['A','B'], f.wilcoxon_p),
                ('Kruskal', ['A','B'], f.kruskal),
                ('Kruskal p', ['A','B'], f.kruskal_p),
                ]
                
    feature_names = [name for (name, dummy1, dummy2) in features]
    
    print("Reading in the training data")
    train = data_io.read_train_pairs()

    print("Extracting features from training data")
    train_features = f.apply_features(train, features)
    
    print("Reading in the validation data")
    valid = data_io.read_valid_pairs()

    print("Extracting features from validation data")
    valid_features = f.apply_features(valid, features)
    
    # Concatenate features
    all_features = train_features
    all_features.update(valid_features)
    
    print("Writing feature file")
    data_io.write_real_features('corrs', all_features, feature_names)
    
if __name__=="__main__":
    main()
