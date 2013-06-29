import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col=0))

def read_train_target():
    path = get_paths()["train_target_path"]
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def read_train_info():
    path = get_paths()["train_info_path"]
    return pd.read_csv(path, index_col=0)

def read_ensemble_train_pairs():
    train_path = get_paths()["ensemble_train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col=0))

def read_ensemble_train_target():
    path = get_paths()["ensemble_train_target_path"]
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def read_ensemble_train_info():
    path = get_paths()["ensemble_train_info_path"]
    return pd.read_csv(path, index_col=0)

def read_valid_pairs():
    valid_path = get_paths()["valid_pairs_path"]
    return parse_dataframe(pd.read_csv(valid_path, index_col=0))

def read_valid_info():
    path = get_paths()["valid_info_path"]
    return pd.read_csv(path, index_col=0)

def read_solution():
    solution_path = get_paths()["solution_path"]
    return pd.read_csv(solution_path, index_col=0)

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path, index_col=0)

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)
    
def write_real_features(filename, IDs, values, feature_names):
    real_feature_path = get_paths()["real_feature_path"]
    writer = csv.writer(open(os.path.join(real_feature_path, filename + '.csv'), "w"), lineterminator="\n", delimiter=',')
    rows = [[IDs[i]] + list(values[i]) for i in range(len(IDs))]
    writer.writerow(["SampleID"] + feature_names)
    writer.writerows(rows)
