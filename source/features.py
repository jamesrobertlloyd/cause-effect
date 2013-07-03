import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import gmean
from scipy.stats import moment
import scipy.stats

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

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

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
