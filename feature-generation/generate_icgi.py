import numpy as np
import os.path

import data_io
import features as f

def main(overwrite=False):

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'icgi.csv')):
        print 'Feature file already exists - not overwriting'
        return

    features = [('ICGI entropy AB', ['A','B'], f.icgi_entropy_AB),
                ('ICGI entropy BA', ['A','B'], f.icgi_entropy_BA),
                ('ICGI entropy diff', 'derived', 'output[key][0] - output[key][1]'),
                ('ICGI slope AB', ['A','B'], f.icgi_slope_AB),
                ('ICGI slope BA', ['A','B'], f.icgi_slope_BA),
                ('ICGI slope diff', 'derived', 'output[key][3] - output[key][4]')]#,
                #('ICGI entropy AB PIT', ['A','B'], f.icgi_entropy_AB_PIT),
                #('ICGI entropy BA PIT', ['A','B'], f.icgi_entropy_BA_PIT),
                #('ICGI entropy diff PIT', 'derived', 'output[key][6] - output[key][7]'),
                #('ICGI slope AB PIT', ['A','B'], f.icgi_slope_AB_PIT),
                #('ICGI slope BA PIT', ['A','B'], f.icgi_slope_BA_PIT),
                #('ICGI slope diff PIT', 'derived', 'output[key][9] - output[key][10]')]
                
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
    data_io.write_real_features('icgi', all_features, feature_names)
    
if __name__=="__main__":
    main()
