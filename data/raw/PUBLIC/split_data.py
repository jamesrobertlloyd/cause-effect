import random

# Load training inputs and outputs

with open('CEdata_train_pairs.csv', 'r') as pairs_data_file:
    pairs_header = pairs_data_file.readline()
    pairs_body = pairs_data_file.readlines()
with open('CEdata_train_target.csv', 'r') as target_data_file:
    target_header = target_data_file.readline()
    target_body = target_data_file.readlines()
with open('CEdata_train_publicinfo.csv', 'r') as info_data_file:
    info_header = info_data_file.readline()
    info_body = info_data_file.readlines()
    
# Create a random partition

random.seed(0)
perm = range(len(pairs_body))
random.shuffle(perm)
train_indices = perm[:int(round(len(pairs_body) / 2.0))]
test_indices  = perm[int(round(len(pairs_body) / 2.0)):]

# Save shuffled files

with open('CEdata_train_reduced_pairs.csv', 'w') as pairs_data_file:
    pairs_data_file.write(pairs_header + ''.join([pairs_body[index] for index in train_indices]))
with open('CEdata_train_ensemble_pairs.csv', 'w') as pairs_data_file:
    pairs_data_file.write(pairs_header + ''.join([pairs_body[index] for index in test_indices]))
    
with open('CEdata_train_reduced_target.csv', 'w') as target_data_file:
    target_data_file.write(target_header + ''.join([target_body[index] for index in train_indices]))
with open('CEdata_train_ensemble_target.csv', 'w') as target_data_file:
    target_data_file.write(target_header + ''.join([target_body[index] for index in test_indices]))
    
with open('CEdata_train_reduced_publicinfo.csv', 'w') as info_data_file:
    info_data_file.write(info_header + ''.join([info_body[index] for index in train_indices]))
with open('CEdata_train_ensemble_publicinfo.csv', 'w') as info_data_file:
    info_data_file.write(info_header + ''.join([info_body[index] for index in test_indices]))
