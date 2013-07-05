import json
import numpy as np
import os
from counter import Progress

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    with open(train_path, 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    pairs = {}
    prog = Progress(len(pairs_body))
    for line in pairs_body:
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        pairs[line.split(',')[0]] = (A, B)
        prog.tick()
    prog.done()
    return pairs

def read_valid_pairs():
    valid_path = get_paths()["valid_pairs_path"]
    with open(valid_path, 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    pairs = {}
    prog = Progress(len(pairs_body))
    for line in pairs_body:
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        pairs[line.split(',')[0]] = (A, B)
        prog.tick()
    prog.done()
    return pairs
    
def write_real_features(filename, feature_list, feature_names):
    real_feature_path = get_paths()["real_feature_path"]
    outfile = open(os.path.join(real_feature_path, filename + '.csv'), "w")
    lines = []
    lines.append(','.join(['SampleID'] + feature_names) + '\n')
    for (sample_id, features) in feature_list.iteritems():
        lines.append(','.join([sample_id] + [str(f) for f in features]) + '\n')
    outfile.writelines(lines)
    outfile.close()
