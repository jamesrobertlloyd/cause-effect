import numpy as np
import os

from counter import Progress

def combine(overwrite=False):
    # Loads the various data sources and concatenates them
    
    if (not overwrite) and os.path.exists('training/CEdata_train_pairs.csv') :
        print 'Output already exists - not overwriting'
        return

    with open('raw/CEfinal_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    with open('raw/CEdata_sup1_train_pairs.csv', 'r') as pairs_data_file:
        dummy = pairs_data_file.readline()
        pairs_sup1_body = pairs_data_file.readlines()
    with open('raw/CEdata_sup2_train_pairs.csv', 'r') as pairs_data_file:
        dummy = pairs_data_file.readline()
        pairs_sup2_body = pairs_data_file.readlines()
    with open('raw/CEdata_sup3_train_pairs.csv', 'r') as pairs_data_file:
        dummy = pairs_data_file.readline()
        pairs_sup3_body = pairs_data_file.readlines()
        
    # Concatenate - making row names unique

    pairs_body_non_unique = pairs_body + pairs_sup1_body + pairs_sup2_body + pairs_sup3_body
    pairs_body = []
    for (i, line) in enumerate(pairs_body_non_unique):
        pairs_body.append(','.join(['train%d' % (i+1)] + line.split(',')[1:]))
        
    with open('raw/CEfinal_train_target.csv', 'r') as target_data_file:
        target_header = target_data_file.readline()
        target_body = target_data_file.readlines()
    with open('raw/CEdata_sup1_train_target.csv', 'r') as target_data_file:
        dummy = target_data_file.readline()
        target_sup1_body = target_data_file.readlines()
    with open('raw/CEdata_sup2_train_target.csv', 'r') as target_data_file:
        dummy = target_data_file.readline()
        target_sup2_body = target_data_file.readlines()
    with open('raw/CEdata_sup3_train_target.csv', 'r') as target_data_file:
        dummy = target_data_file.readline()
        target_sup3_body = target_data_file.readlines()
        
    # Concatenate - making row names unique

    target_body_non_unique = target_body + target_sup1_body + target_sup2_body + target_sup3_body
    target_body = []
    for (i, line) in enumerate(target_body_non_unique):
        target_body.append(','.join(['train%d' % (i+1)] + line.split(',')[1:]))
        
    with open('raw/CEfinal_train_publicinfo.csv', 'r') as info_data_file:
        info_header = info_data_file.readline()
        info_body = info_data_file.readlines()    
    with open('raw/CEdata_sup1_train_publicinfo.csv', 'r') as info_data_file:
        dummy = info_data_file.readline()
        info_sup1_body = info_data_file.readlines()    
    with open('raw/CEdata_sup2_train_publicinfo.csv', 'r') as info_data_file:
        dummy = info_data_file.readline()
        info_sup2_body = info_data_file.readlines()  
    with open('raw/CEdata_sup3_train_publicinfo.csv', 'r') as info_data_file:
        dummy = info_data_file.readline()
        info_sup3_body = info_data_file.readlines()
        
    # Concatenate - making row names unique

    info_body_non_unique = info_body + info_sup1_body + info_sup2_body + info_sup3_body
    info_body = []
    for (i, line) in enumerate(info_body_non_unique):
        info_body.append(','.join(['train%d' % (i+1)] + line.split(',')[1:]))

    # Save files

    with open('training/CEdata_train_pairs.csv', 'w') as pairs_data_file:
        pairs_data_file.write(pairs_header + ''.join(pairs_body))
        
    with open('training/CEdata_train_target.csv', 'w') as target_data_file:
        target_data_file.write(target_header + ''.join(target_body))
        
    with open('training/CEdata_train_publicinfo.csv', 'w') as info_data_file:
        info_data_file.write(info_header + ''.join(info_body))

def reverse_it(overwrite=False):
    
    if (not overwrite) and os.path.exists('training-reversed/CEdata_train_pairs.csv') :
        print 'Output already exists - not overwriting'
        return
        
    # Open info
    with open('training/CEdata_train_publicinfo.csv', 'r') as info_data_file:
        info_header = info_data_file.readline()
        info_body = info_data_file.readlines()
    # Reverse it (no change)
    original_length = len(info_body)
    for i in range(original_length):
        info_body.append(','.join(['train%d' % (len(info_body)+1)] + info_body[i].split(',')[1:]))
        info_body.append(','.join(['train%d' % (len(info_body)+1)] + info_body[i].split(',')[1:]))
        info_body.append(','.join(['train%d' % (len(info_body)+1)] + info_body[i].split(',')[1:]))
        
    # Open targets
    with open('training/CEdata_train_target.csv', 'r') as target_data_file:
        target_header = target_data_file.readline()
        target_body = target_data_file.readlines()
    # Reverse it - no change
    original_length = len(target_body)
    for i in range(original_length):
        target_body.append(','.join(['train%d' % (len(target_body)+1)] + target_body[i].split(',')[1:]))
        target_body.append(','.join(['train%d' % (len(target_body)+1)] + target_body[i].split(',')[1:]))
        target_body.append(','.join(['train%d' % (len(target_body)+1)] + target_body[i].split(',')[1:]))
        
    # Open pairs
    with open('training/CEdata_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        # Write reversed lines to temporary file
        with open('temp.csv', 'w') as temp_file:
            temp_file.write(pairs_header)
            prog = Progress(original_length * 5)
            for line in pairs_data_file:
                A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
                B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
                if set(A) == set([0, 1]):
                    A_reversed = 1 - A
                else:
                    A_reversed = 2 * np.mean(A) - A
                if set(B) == set([0, 1]):
                    B_reversed = 1 - B
                else:
                    B_reversed = 2 * np.mean(B) - B
                temp_file.write(','.join(['dummy-id'] + [' '.join(str(a) for a in A_reversed)] + [' '.join(str(b) for b in B)]) + '\n')
                temp_file.write(','.join(['dummy-id'] + [' '.join(str(a) for a in A)] + [' '.join(str(b) for b in B_reversed)]) + '\n')
                temp_file.write(','.join(['dummy-id'] + [' '.join(str(a) for a in A_reversed)] + [' '.join(str(b) for b in B_reversed)]) + '\n')
                prog.tick()
                
    # Concatenate original pairs and temporary file 
    with open('training-reversed/CEdata_train_pairs.csv', 'w') as pairs_data_file:
        pairs_data_file.write(pairs_header)
        i = 1
        for file_name in ['training/CEdata_train_pairs.csv', 'temp.csv']:
            with open(file_name, 'r') as input_file:
                input_file.readline()
                for line in input_file:
                    pairs_data_file.write(','.join(['train%d' % i] + line.split(',')[1:]))
                    prog.tick()
                    i += 1
    prog.done()    
    os.remove('temp.csv') 
    # Save other files
        
    with open('training-reversed/CEdata_train_target.csv', 'w') as target_data_file:
        target_data_file.write(target_header + ''.join(target_body))
        
    with open('training-reversed/CEdata_train_publicinfo.csv', 'w') as info_data_file:
        info_data_file.write(info_header + ''.join(info_body))
        
def flip_reverse_it(overwrite=False):
    
    if (not overwrite) and os.path.exists('training-flipped-reversed/CEdata_train_pairs.csv') :
        print 'Output already exists - not overwriting'
        return
        
    # Open info
    with open('training-reversed/CEdata_train_publicinfo.csv', 'r') as info_data_file:
        info_header = info_data_file.readline()
        info_body = info_data_file.readlines()
    # Flip it
    original_length = len(info_body)
    prog = Progress(original_length)
    for i in range(original_length):
        info_body.append(','.join(['train%d' % (len(info_body)+1)] + list(reversed(info_body[i].strip().split(',')[1:]))) + '\n')
        prog.tick()
    prog.done()
        
    # Open targets
    with open('training-reversed/CEdata_train_target.csv', 'r') as target_data_file:
        target_header = target_data_file.readline()
        target_body = target_data_file.readlines()
    # Flip it
    original_length = len(target_body)
    prog = Progress(original_length)
    for i in range(original_length):
        targets = target_body[i].split(',')[1:]
        if targets[0] == '1':
            targets[0] = '-1'
        elif targets[0] == '-1':
            targets[0] = '1'
        if targets[1] == '1\n':
            targets[1] = '2\n'
        elif targets[1] == '2\n':
            targets[1] = '1\n'
        target_body.append(','.join(['train%d' % (len(target_body)+1)] + targets))
        prog.tick()
    prog.done()
    
    # Open pairs
    with open('training-reversed/CEdata_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        with open('temp.csv', 'w') as temp_file:
            temp_file.write(pairs_header)
            prog = Progress(original_length * 3)
            # Save flipped lines to temporary file
            for line in pairs_data_file:
                temp_file.write(','.join(['dummy-id'] + list(reversed(line.strip().split(',')[1:]))) + '\n')
                prog.tick()
        
    # Concatenate original pairs and temporary file 
    with open('training-flipped-reversed/CEdata_train_pairs.csv', 'w') as pairs_data_file:
        pairs_data_file.write(pairs_header)
        i = 1
        for file_name in ['training-reversed/CEdata_train_pairs.csv', 'temp.csv']:
            with open(file_name, 'r') as input_file:
                input_file.readline()
                for line in input_file:
                    pairs_data_file.write(','.join(['train%d' % i] + line.split(',')[1:]))
                    prog.tick()
                    i += 1
    prog.done()    
    os.remove('temp.csv')

    # Save other files
        
    with open('training-flipped-reversed/CEdata_train_target.csv', 'w') as target_data_file:
        target_data_file.write(target_header + ''.join(target_body))
        
    with open('training-flipped-reversed/CEdata_train_publicinfo.csv', 'w') as info_data_file:
        info_data_file.write(info_header + ''.join(info_body))
        
def flip_it(overwrite=True):
    
    if (not overwrite) and os.path.exists('training-flipped/CEdata_train_pairs.csv') :
        print 'Output already exists - not overwriting'
        return
        
    # Open info
    with open('training/CEdata_train_publicinfo.csv', 'r') as info_data_file:
        info_header = info_data_file.readline()
        info_body = info_data_file.readlines()
    # Flip it
    original_length = len(info_body)
    prog = Progress(original_length)
    for i in range(original_length):
        info_body.append(','.join(['train%d' % (len(info_body)+1)] + list(reversed(info_body[i].strip().split(',')[1:]))) + '\n')
        prog.tick()
    prog.done()
        
    # Open targets
    with open('training/CEdata_train_target.csv', 'r') as target_data_file:
        target_header = target_data_file.readline()
        target_body = target_data_file.readlines()
    # Flip it
    original_length = len(target_body)
    prog = Progress(original_length)
    for i in range(original_length):
        targets = target_body[i].split(',')[1:]
        if targets[0] == '1':
            targets[0] = '-1'
        elif targets[0] == '-1':
            targets[0] = '1'
        if targets[1] == '1\n':
            targets[1] = '2\n'
        elif targets[1] == '2\n':
            targets[1] = '1\n'
        target_body.append(','.join(['train%d' % (len(target_body)+1)] + targets))
        prog.tick()
    prog.done()
    
    # Open pairs
    with open('training/CEdata_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        with open('temp.csv', 'w') as temp_file:
            temp_file.write(pairs_header)
            prog = Progress(original_length * 3)
            # Save flipped lines to temporary file
            for line in pairs_data_file:
                temp_file.write(','.join(['dummy-id'] + list(reversed(line.strip().split(',')[1:]))) + '\n')
                prog.tick()
        
    # Concatenate original pairs and temporary file 
    with open('training-flipped/CEdata_train_pairs.csv', 'w') as pairs_data_file:
        pairs_data_file.write(pairs_header)
        i = 1
        for file_name in ['training/CEdata_train_pairs.csv', 'temp.csv']:
            with open(file_name, 'r') as input_file:
                input_file.readline()
                for line in input_file:
                    pairs_data_file.write(','.join(['train%d' % i] + line.split(',')[1:]))
                    prog.tick()
                    i += 1

    # Save other files
        
    with open('training-flipped/CEdata_train_target.csv', 'w') as target_data_file:
        target_data_file.write(target_header + ''.join(target_body))
        
    with open('training-flipped/CEdata_train_publicinfo.csv', 'w') as info_data_file:
        info_data_file.write(info_header + ''.join(info_body))
    
if __name__=="__main__":
    print('Concatenating data files to form training data')
    combine()
    print('Producing flipped data set')
    flip_it()
    print('Producing reversed data set')
    reverse_it()
    print('Producing flipped and reversed data set')
    flip_reverse_it()
