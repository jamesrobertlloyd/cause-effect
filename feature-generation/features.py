import numpy as np
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import gmean
from scipy.stats import moment
import scipy.stats
from counter import Progress
            
def apply_features(data, features):
    #### TODO - can I be made more efficient?
    prog = Progress(len(data))
    output = {key : np.zeros(len(features)) for key in data.keys()}
    for (key, (A, B)) in data.iteritems():
        for (j, (name, variable, f)) in enumerate(features):
            if variable == 'A':
                value = f(A)
            elif variable == 'B':
                value = f(B)
            elif variable == 'derived':
                value = eval(f)
            else:
                value = f(A, B)
            output[key][j] = value
        prog.tick()
    prog.done()
    return output

def identity(x):
    return x

def count_unique(x):
    return len(set(x))

def unique_ratio(x):
    return count_unique(x)/float(len(x))

def mean(x):
    return np.mean(x)

def median(x):
    return scipy.stats.cmedian(x)

def sd(x):
    return np.std(x)

def fkurtosis(x):
    return kurtosis(x)

def fkurtosis_diff(x, y):
    return kurtosis(x) - kurtosis(y)

def fkurtosis_ratio(x, y):
    if kurtosis(y) == 0:
        return 0
    else:
        return kurtosis(x) / kurtosis(y)

def fskew(x):
    return skew(x)

def fskew_diff(x, y):
    return skew(x) - skew(y)

def fskew_ratio(x, y):
    if skew(y) == 0:
        return 0
    else:
        return skew(x) / skew(y)

def fgmean(x):
    return gmean(x)
    
def standard_moment_5(x):
    return moment(x, 5) / (sd(x) ** 5)
    
def standard_moment_diff_5(x, y):
    return standard_moment_5(x) - standard_moment_5(y)
    
def standard_moment_ratio_5(x, y):
    if standard_moment_5(y) == 0:
        return 0
    else:
        return standard_moment_5(x) / standard_moment_5(y)
    
def standard_moment_6(x):
    return moment(x, 6) / (sd(x) ** 6)
    
def standard_moment_diff_6(x, y):
    return standard_moment_6(x) - standard_moment_6(y)
    
def standard_moment_ratio_6(x, y):
    if standard_moment_6(y) == 0:
        return 0
    else:
        return standard_moment_6(x) / standard_moment_6(y)
    
def standard_moment_7(x):
    return moment(x, 7) / (sd(x) ** 7)
    
def standard_moment_diff_7(x, y):
    return standard_moment_7(x) - standard_moment_7(y)
    
def standard_moment_ratio_7(x, y):
    if standard_moment_7(y) == 0:
        return 0
    else:
        return standard_moment_7(x) / standard_moment_7(y)
    
def standard_moment_8(x):
    return moment(x, 8) / (sd(x) ** 8)
    
def standard_moment_diff_8(x, y):
    return standard_moment_8(x) - standard_moment_8(y)
    
def standard_moment_ratio_8(x, y):
    if standard_moment_8(y) == 0:
        return 0
    else:
        return standard_moment_8(x) / standard_moment_8(y)
    
def standard_moment_9(x):
    return moment(x, 9) / (sd(x) ** 9)
    
def standard_moment_diff_9(x, y):
    return standard_moment_9(x) - standard_moment_9(y)
    
def standard_moment_ratio_9(x, y):
    if standard_moment_9(y) == 0:
        return 0
    else:
        return standard_moment_9(x) / standard_moment_9(y)

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    
    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);

    return hx

def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)

def entropy_ratio(x, y):
    ex = normalized_entropy(x) 
    ey = normalized_entropy(y)
    if ey == 0:
	    # upps
	    ey = 0.0000000000001
    return (ex/ey)

def correlation(x, y):
    return pearsonr(x, y)[0]

def rcorrelation(x, y):
    return spearmanr(x, y)[0]

def rcorrelation_magnitude(x, y):
    return abs(spearmanr(x, y)[0])

def correlation_magnitude(x, y):
    return abs(correlation(x, y))

def Pearson_Spearman_diff(x, y):
    return correlation(x, y) - rcorrelation(x, y)

def Pearson_Spearman_abs_diff(x, y):
    return correlation_magnitude(x, y) - rcorrelation_magnitude(x, y)

def Pearson_Spearman_ratio(x, y):
    if rcorrelation(x, y) == 0:
        0
    else:
        return correlation(x, y) / rcorrelation(x, y)
     
def kendall(x, y):
    return scipy.stats.kendalltau(x, y)[0]
     
def kendall_p(x, y):
    return scipy.stats.kendalltau(x, y)[1]
     
def mannwhitney(x, y):
    return scipy.stats.mannwhitneyu(x, y)[0]
     
def mannwhitney_p(x, y):
    return scipy.stats.mannwhitneyu(x, y)[1]
     
def wilcoxon(x, y):
    return scipy.stats.wilcoxon(x, y)[0]
     
def wilcoxon_p(x, y):
    return scipy.stats.wilcoxon(x, y)[1]
     
def kruskal(x, y):
    return scipy.stats.kruskal(x, y)[0]
     
def kruskal_p(x, y):
    return scipy.stats.kruskal(x, y)[1]
    
def injectivity(A, B, image_size):
    # Converts pairs into scatter plot bitmap image
    image = np.zeros([image_size, image_size])
    min_A = min(A)
    min_B = min(B)
    max_A = max(A)
    max_B = max(B)
    for (a, b) in zip(A, B):
        # Record the data point in the square it falls in
        image[np.floor(min(image_size - 1, (a - min_A) * image_size / (max_A - min_A))), min(image_size - 1, np.floor((b - min_B) * image_size / (max_B - min_B)))] += 1
    # Binarise
    image = image >= 1
    # Attempt to count multiple values
    count_B = 0
    for i in range(image_size):
        state = 0 # Waiting for first data point
        for j in range(image_size):
            if state == 0:
                if image[i,j]:
                    state = 1 # Seen a data point
            elif state == 1:
                if not image[i,j]:
                    state = 2 # Seen the start of a potential gap
            elif state == 2:
                if image[i,j]:
                    count_B += 1 # Seen a gap
                    state = 1 # Wait for another gap to start
    count_A = 0
    for j in range(image_size):
        state = 0 # Waiting for first data point
        for i in range(image_size):
            if state == 0:
                if image[i,j]:
                    state = 1 # Seen a data point
            elif state == 1:
                if not image[i,j]:
                    state = 2 # Seen the start of a potential gap
            elif state == 2:
                if image[i,j]:
                    count_A += 1 # Seen a gap
                    state = 1 # Wait for another gap to start
    # Return a measure of gappy ness of the data
    return (1.0 + (count_A * 1.0 / image_size)) / (1.0 + (count_B * 1.0 / image_size))
    
def injectivity_10(A, B):
    return injectivity(A, B, 10)
    
def injectivity_20(A, B):
    return injectivity(A, B, 10)
    
def injectivity_40(A, B):
    return injectivity(A, B, 10)
