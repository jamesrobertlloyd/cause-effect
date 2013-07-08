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

import numpy as num
import itertools
from dbn import *
import sys
import matplotlib.pyplot as plt
import matplotlib

def numMistakes(targetsMB, outputs):
    if not isinstance(outputs, num.ndarray):
        outputs = outputs.as_numpy_array()
    if not isinstance(targetsMB, num.ndarray):
        targetsMB = targetsMB.as_numpy_array()
    return num.sum(outputs.argmax(1) != targetsMB.argmax(1))

def sampleMinibatch(mbsz, inps, targs):
    idx = num.random.randint(inps.shape[0], size=(mbsz,))
    return inps[idx], targs[idx]

def main(dropout=False):
    mbsz = 64 # Size of minibatch
    layerSizes = [784, 512, 512, 10] # 28 x 28 visible images, 512 hidden, 512 hidden, 10 labels
    scales = [0.05 for i in range(len(layerSizes)-1)] # Dunno
    fanOuts = [None for i in range(len(layerSizes)-1)] # Restricts number of incoming links (viewing net as visible -> hidden)
    learnRate = 0.03 # Not sure
    epochs = 20 # 20
    mbPerEpoch = 100 # 10000 # int(num.ceil(60000./mbsz)) # Number of mini-batches per epoch
    
    f = num.load("mnist.npz")
    trainInps = f['trainInps']/255.
    testInps = f['testInps']/255.
    trainTargs = f['trainTargs']
    testTargs = f['testTargs']

    assert(trainInps.shape == (60000, 784))
    assert(trainTargs.shape == (60000, 10))
    assert(testInps.shape == (10000, 784))
    assert(testTargs.shape == (10000, 10))

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
        net.learnRates = [2.0 for unused in net.learnRates]  
    else:
        net.learnRates = [0.4 for unused in net.learnRates]      
            
    if dropout:
        pass
        #mbPerEpoch = mbPerEpoch * mbsz / 10
        #mbsz = 10
        #net.learnRates = [0.01 for unused in net.learnRates] # Set the learning rate to be equal
    
    # Fine tuning
    
    epochs = 50
    
    for ep, (trCE, trEr) in enumerate(net.fineTune(mbStream, epochs, mbPerEpoch, numMistakes, True, dropout)):
        print 'Fine tuning Epoch %d, trCE = %s, trEr = %s' % (ep, trCE, trEr)
        
    # Try something
    
    #i = num.random.randint(testInps.shape[0])
    #imshow(num.reshape(testInps[i], (28, 28)), cmap = matplotlib.cm.Greys)
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
    
    return net

if __name__ == "__main__":
    main()
