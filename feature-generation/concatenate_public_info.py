import os.path

import data_io

with open(os.path.join(data_io.get_paths()["real_feature_path"], 'publicinfo.csv'), 'w') as publicinfo_file:
    with open(data_io.get_paths()["train_publicinfo_path"], 'r') as train_file:
        publicinfo_file.write(train_file.read())
    with open(data_io.get_paths()["valid_publicinfo_path"], 'r') as valid_file:
        valid_file.readline()
        publicinfo_file.write(valid_file.read())
