import numpy as np
import os.path

import data_io
import features as f

import oct2py as op

def main(overwrite=False):

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'add.noise.csv')):
        print 'Feature file already exists - not overwriting'
        return

    features = [('Additive noise model AB', ['A','B'], f.add_noise_model_AB),
                ('Additive noise model BA', ['A','B'], f.add_noise_model_BA)]
                
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
    data_io.write_real_features('add_noise', all_features, feature_names)
    
if __name__=="__main__":
    main()
