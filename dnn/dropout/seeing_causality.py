"""
 Copyright (c) 2011,2012 George Dahl

 Permission is hereby granted, free of charge, to any person  obtaining
 a copy of this software and associated documentation  files (the
 "Software"), to deal in the Software without  restriction, including
 without limitation the rights to use,  copy, modify, merge, publish,
 distribute, sublicense, and/or sell  copies of the Software, and to
 permit persons to whom the  Software is furnished to do so, subject
 to the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.  THE
 SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR  OTHER DEALINGS IN THE
 SOFTWARE.
 
 Edited by James Robert Lloyd 26 December 2012
"""

import numpy as np
import itertools
from dbn import *
import sys
import matplotlib.pyplot as plt
import matplotlib
from counter import Progress
import scipy.stats
import pickle

def numMistakes(targetsMB, outputs):
    if not isinstance(outputs, np.ndarray):
        outputs = outputs.as_numpy_array()
    if not isinstance(targetsMB, np.ndarray):
        targetsMB = targetsMB.as_numpy_array()
    return np.sum(outputs.argmax(1) != targetsMB.argmax(1))

def sampleMinibatch(mbsz, inps, targs):
    idx = np.random.randint(inps.shape[0], size=(mbsz,))
    return inps[idx], targs[idx]
    
def pairs_to_image(A, B, image_size):
    # Take PIT of inputs
    A = scipy.stats.norm.cdf(scipy.stats.zscore(A))
    B = scipy.stats.norm.cdf(scipy.stats.zscore(B))
    # Converts pairs into scatter plot bitmap image
    image = np.zeros([image_size, image_size])
    min_A = min(A)
    min_B = min(B)
    max_A = max(A)
    max_B = max(B)
    for (a, b) in zip(A, B):
        # Record the data point in the square it falls in
        image[np.floor(min(image_size - 1, (a - min_A) * image_size / (max_A - min_A))), min(image_size - 1, np.floor((b - min_B) * image_size / (max_B - min_B)))] += 1
    # Normalise
    image = 1.0 * image / len(A)
    # Remove mean
    image = image - np.mean(image)
    # Truncate to 3 std and scale to -1 to 1
    std = 3 * np.std(image)
    image = np.clip(image, -std, std) / std;
    # Rescale from [-1,1] to [0.1,0.9]
    image = (image + 1) * 0.4 + 0.1;
    return image.ravel()
    
def load_stuff():
    net = pickle.load(open('net.p', 'rb'))
    trainInps = pickle.load(open('trainInps.p', 'rb'))
    trainTargs = pickle.load(open('trainTargs.p', 'rb'))
    testInps = pickle.load(open('testInps.p', 'rb'))
    testTargs = pickle.load(open('testTargs.p', 'rb'))

def main(dropout=False):
    mbsz = 1024 # Size of minibatch
    image_size = 20;
    layerSizes = [image_size ** 2, 512, 512, 2] # 28 x 28 visible images, 512 hidden, 512 hidden, 2 labels
    scales = [0.05 for i in range(len(layerSizes)-1)] # Dunno
    fanOuts = [None for i in range(len(layerSizes)-1)] # Restricts number of incoming links (viewing net as visible -> hidden)
    learnRate = 0.02 # Not sure
    epochs = 100 # 20
    mbPerEpoch = 500 # 10000 # int(num.ceil(60000./mbsz)) # Number of mini-batches per epoch
    
    print('Loading pairs data')
    with open('../data/training-flipped/CEdata_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    pairs = []
    prog = Progress(len(pairs_body))
    for line in pairs_body:
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        pairs.append((A, B))
        prog.tick()
    prog.done()
    print('Converting to images')
    Inps = np.zeros([len(pairs), image_size ** 2])
    prog = Progress(len(pairs))
    for (i, (A, B)) in enumerate(pairs):
        Inps[i,:] = pairs_to_image(A, B, image_size)
        prog.tick()
    prog.done()  
    print('Loading targets')
    Targs = np.zeros(len(pairs))
    with open('../data/training-flipped/CEdata_train_target.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    Targs = np.array(list(zip([1.0 if line.split(',')[1] == '1' else 0.0 for line in pairs_body], [0.0 if line.split(',')[1] == '1' else 1.0 for line in pairs_body])))
    print('Splitting data')
    trainInps = Inps[0:28000,:]
    trainTargs = Targs[0:28000,:]
    testInps = Inps[28000:32000,:]
    testTargs = Targs[28000:32000,:]
    print('Doing DNN stuff')
    
    #f = np.load("gdahl/mnist.npz")
    #trainInps = f['trainInps']/255.
    #testInps = f['testInps']/255.
    #trainTargs = f['trainTargs']
    #testTargs = f['testTargs']

    #assert(trainInps.shape == (60000, 784))
    #assert(trainTargs.shape == (60000, 10))
    #assert(testInps.shape == (10000, 784))
    #assert(testTargs.shape == (10000, 10))

    # A generator of minbatches
    mbStream = (sampleMinibatch(mbsz, trainInps, trainTargs) for unused in itertools.repeat(None))
    
    if dropout:
        net = buildDBN(layerSizes, scales, fanOuts, Softmax(), realValuedVis=False, dropouts = [0.2,0.5,0.5])
    else:
        net = buildDBN(layerSizes, scales, fanOuts, Softmax(), realValuedVis=False)
    net.learnRates = [learnRate for unused in net.learnRates] # Set the learning rate to be equal
    net.L2Costs = [0 for unused in net.L2Costs] # Presumably a lack of regularisation?
    net.nestCompare = True #this flag existing is a design flaw that I might address later, for now always set it to True
    
    # Pre-training
    for layer in range(len(layerSizes)-2):
        for (epoch, state) in enumerate(net.preTrainIth(layer, mbStream, epochs, mbPerEpoch)):
            print 'Layer %d Epoch %d State = %s' % (layer, epoch+1, state)
    
    if dropout:
        net.learnRates = [0.2 for unused in net.learnRates]  
    else:
        net.learnRates = [0.4 for unused in net.learnRates]      
            
    if dropout:
        pass
        #mbPerEpoch = mbPerEpoch * mbsz / 10
        #mbsz = 10
        #net.learnRates = [0.01 for unused in net.learnRates] # Set the learning rate to be equal
    
    # Fine tuning
    
    epochs = 1000000
    
    for ep, (trCE, trEr) in enumerate(net.fineTune(mbStream, epochs, mbPerEpoch, numMistakes, True, dropout)):
        print 'Fine tuning Epoch %d, trCE = %s, trEr = %s' % (ep, trCE, trEr)
        
    # Try something
    
    #i = np.random.randint(testInps.shape[0])
    #imshow(np.reshape(testInps[i], (28, 28)), cmap = matplotlib.cm.Greys)
    #predictions = numpyify(net.fprop(testInps[i]) * 100)[0]
    #for (i, p) in reversed(sorted(enumerate(predictions), key=lambda x:x[1])):
    #    print 'Digit %d Probability = %2.0f' % (i, p)
        
    # Determine testing error rate
    
    num_correct = 0
    for (i, (testInp, testTarg)) in enumerate(zip(testInps, testTargs)):
        predictions = [x * 100 for x in net.fprop(testInp)[0]]
        if testTarg[predictions.index(max(predictions))]:
            num_correct += 1
        if (i % 500) == 0:
            print '.',
            sys.stdout.flush()
            
    print '\nPercentage correct = %2.2f%%' % (num_correct * 100.0 / testTargs.shape[0])
    
    print('Loading validation data')
    with open('../data/validation/CEfinal_valid_pairs.csv', 'r') as valid_data_file:
        valid_header = valid_data_file.readline()
        valid_body = valid_data_file.readlines()
    valid = []
    prog = Progress(len(valid_body))
    for line in valid_body:
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        valid.append((A, B))
        prog.tick()
    prog.done()
    print('Converting to images')
    validInps = np.zeros([len(valid), image_size ** 2])
    prog = Progress(len(valid))
    for (i, (A, B)) in enumerate(valid):
        validInps[i,:] = pairs_to_image(A, B, image_size)
        prog.tick()
    prog.done()  
    
    predictions = []
    for (i, validInp) in enumerate(validInps):
        predictions.append(net.fprop(validInp)[0][0])
        
    i = np.random.randint(len(valid))
    print predictions[i]    
    imshow(np.reshape(validInps[i], (image_size, image_size)))
    
    valid_row_names = ['valid%d' % i for i in range(1, 4050+1, 1)]
    lines = ['SampleID,Target\n'] + [valid_row_names[i] + ',' + str(p) + '\n' for (i, p) in enumerate(predictions)]
    with open('predictions.csv', 'w') as outfile:
        outfile.writelines(lines)
    
    return net
    

if __name__ == "__main__":
    main()
