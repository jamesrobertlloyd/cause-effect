import numpy as np
import os.path

import data_io
import features as f

def main(overwrite=False):

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'high_order_moments.csv')):
        print 'Feature file already exists - not overwriting'
        return

    features = [('Moment 5 A', 'A', f.standard_moment_5),
                ('Moment 5 B', 'B', f.standard_moment_5),
                ('Moment 5 diff', 'derived', 'output[key][0] - output[key][1]'),
                ('Moment 5 ratio', 'derived', 'output[key][0] / output[key][1] if not output[key][1] == 0 else output[key][0] / 0.000001'),
                ('Moment 6 A', 'A', f.standard_moment_6),
                ('Moment 6 B', 'B', f.standard_moment_6),
                ('Moment 6 diff', 'derived', 'output[key][4] - output[key][5]'),
                ('Moment 6 ratio', 'derived', 'output[key][4] / output[key][5] if not output[key][5] == 0 else output[key][4] / 0.000001'),
                ('Moment 7 A', 'A', f.standard_moment_7),
                ('Moment 7 B', 'B', f.standard_moment_7),
                ('Moment 7 diff', 'derived', 'output[key][8] - output[key][9]'),
                ('Moment 7 ratio', 'derived', 'output[key][8] / output[key][9] if not output[key][9] == 0 else output[key][8] / 0.000001'),
                ('Moment 8 A', 'A', f.standard_moment_8),
                ('Moment 8 B', 'B', f.standard_moment_8),
                ('Moment 8 diff', 'derived', 'output[key][12] - output[key][13]'),
                ('Moment 8 ratio', 'derived', 'output[key][12] / output[key][13] if not output[key][13] == 0 else output[key][12] / 0.000001'),
                ('Moment 9 A', 'A', f.standard_moment_9),
                ('Moment 9 B', 'B', f.standard_moment_9),
                ('Moment 9 diff', 'derived', 'output[key][16] - output[key][17]'),
                ('Moment 9 ratio', 'derived', 'output[key][16] / output[key][17] if not output[key][17] == 0 else output[key][16] / 0.000001')]
                
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
    data_io.write_real_features('high_order_moments', all_features, feature_names)
    
if __name__=="__main__":
    main()
