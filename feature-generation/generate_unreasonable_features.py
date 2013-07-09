import data_io as data_io
import features as f
import numpy as np

def main():

    #### TODO - sequential processing of data would significantly reduce memory demands
    
    if (not overwrite) and os.path.exists(os.path.join(data_io.get_paths()["real_feature_path"], 'unreasonable_features.csv')):
        print 'Feature file already exists - not overwriting'
        return
        
    features = [('Number of Samples', 'A', len),
                ('Max A', 'A', max),
                ('Max B', 'B', max),
                ('Min A', 'A', min),
                ('Min B', 'B', min),
                ('Mean A', 'A', f.mean),
                ('Mean B', 'B', f.mean),
                ('Median A', 'A', f.median),
                ('Median B', 'B', f.median),
                ('Sd A', 'A', f.sd),
                ('Sd B', 'B', f.sd)]
                
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
    data_io.write_real_features('unreasonable_features', all_features, feature_names)
    
if __name__=="__main__":
    main()
