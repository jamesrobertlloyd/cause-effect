'''
Various utilities relating to basic pre-processing
'''

import numpy as np

def create_random_predictors():
    '''Creates the example files in the predictors directory'''
    IDs = ['train%d' % i for i in range(1, 7832)] + ['valid%d' % i for i in range(1, 2643)]
    # Create real valued random predictor
    with open('../predictors/real/random_example.csv', 'w') as save_file:
        save_file.write('SampleID,Target\n')
        random_lines = ['%s,%f' % (ID, (0.5 - np.random.random()) * 10) for ID in IDs]
        save_file.write('\n'.join(random_lines))
    # Create 4 class random predictor
    with open('../predictors/4class/random_example.csv', 'w') as save_file:
        save_file.write('SampleID,A->B,B->A,A-B,A|B\n')
        random_lines = ['%s,%f,%f,%f,%f' % ((ID,) + tuple(np.random.dirichlet([1] * 4))) for ID in IDs]
        save_file.write('\n'.join(random_lines))
    # Create 3 class random predictor
    with open('../predictors/3class/random_example.csv', 'w') as save_file:
        save_file.write('SampleID,A->B,B->A,A-B or A|B\n')
        random_lines = ['%s,%f,%f,%f' % ((ID,) + tuple(np.random.dirichlet([1] * 3))) for ID in IDs]
        save_file.write('\n'.join(random_lines))
