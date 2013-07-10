import numpy as np
import os.path

import data_io
import features as f

def main(overwrite=False):

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'injectivity.csv')):
        print 'Feature file already exists - not overwriting'
        return

    features = [('Injectivity 10', ['A','B'], f.injectivity_10),
                ('Injectivity 15', ['A','B'], f.injectivity_15),
                ('Injectivity 20', ['A','B'], f.injectivity_20),
                ('Injectivity 25', ['A','B'], f.injectivity_25),
                ('Injectivity 30', ['A','B'], f.injectivity_30),
                ('Injectivity 35', ['A','B'], f.injectivity_35),
                ('Injectivity 40', ['A','B'], f.injectivity_40)]
                
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
    data_io.write_real_features('injectivity', all_features, feature_names)
    
if __name__=="__main__":
    main()
