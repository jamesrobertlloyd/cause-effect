import data_io
import features as f
import numpy as np

def main():
    features = [('A: Normalized Entropy', 'A', f.normalized_entropy),
                ('B: Normalized Entropy', 'B', f.normalized_entropy),
                ('Pearson R', ['A','B'], f.correlation),
                ('Pearson R Magnitude', 'derived', 'abs(output[key][2])'),
                ('Entropy Difference', 'derived', 'output[key][0] - output[key][1]'),
                ('Entropy Ratio', 'derived', 'output[key][0] / output[key][1] if not output[key][1] == 0 else output[key][0] / 0.000001'),
                ('Spearman rank correlation', ['A','B'], f.rcorrelation),
                ('Spearman rank magnitude', 'derived', 'abs(output[key][6])'),
                ('Kurtosis A', 'A', f.fkurtosis),
                ('Kurtosis B', 'B', f.fkurtosis),
                ('Kurtosis difference', 'derived', 'output[key][8] - output[key][9]'),
                ('Kurtosis ratio', 'derived', 'output[key][8] / output[key][9] if not output[key][9] == 0 else output[key][8] / 0.000001'),
                ('Unique ratio A', 'A', f.unique_ratio),
                ('Unique ratio B', 'B', f.unique_ratio),
                ('Skew A', 'A', f.fskew),
                ('Skew B', 'B', f.fskew),
                ('Skew difference', 'derived', 'output[key][14] - output[key][15]'),
                ('Skew ratio', 'derived', 'output[key][14] / output[key][15] if not output[key][15] == 0 else output[key][14] / 0.000001'),
                ('Pearson - Spearman', 'derived', 'output[key][2] - output[key][6]'),
                ('Abs Pearson - Spearman', 'derived', 'output[key][3] - output[key][7]'),
                ('Pearson / Spearman', 'derived', 'output[key][2] / output[key][6] if not output[key][6] == 0 else output[key][2] / 0.000001')]
                
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
    data_io.write_real_features('reasonable_features', all_features, feature_names)
    
if __name__=="__main__":
    main()
