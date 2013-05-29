#!/usr/bin/env python
# encoding: utf-8
"""
parzen.py

Parzen window evaluator for generative models.

Created by Yann N. Dauphin on 2010-10-02.
Copyright (c) 2010 Lambda Tree Media. All rights reserved.
"""

import sys
import os
import numpy
import cPickle, gzip
import time

import theano
from theano import tensor as T
import image_tiler

def local_contrast_normalization(patches):
    patches = patches.reshape((patches.shape[0], -1))
    patches -= patches.mean(axis=1)[:,None]

    patches_std = numpy.sqrt((patches**2).mean(axis=1))

    min_divisor = (2*patches_std.min() + patches_std.mean()) / 3
    patches /= numpy.maximum(min_divisor, patches_std).reshape((patches.shape[0],1))

    return patches


def log_mean_exp(a):
    max_ = a.max(1)
    
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    x = T.matrix()
    mu = theano.shared(mu)
    
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    
    E = log_mean_exp(-0.5*(a**2).sum(2))
    
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))
    
    return theano.function([x], E - Z)


def numpy_parzen(x, mu, sigma):
    a = ( x[:, None, :] - mu[None, :, :] ) / sigma
    
    def log_mean(i):
        return i.max(1) + numpy.log(numpy.exp(i - i.max(1)[:, None]).mean(1))
    
    return log_mean(-0.5 * (a**2).sum(2)) - mu.shape[1] * numpy.log(sigma * numpy.sqrt(numpy.pi * 2))


def get_nll(x, parzen, batch_size=10):
    inds = range(x.shape[0])
    
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))
    
    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        
        times.append(end-begin)
        
        nlls.extend(nll)
        
        if i % 10 == 0:
            print i, numpy.mean(times), numpy.mean(nlls)
    
    return nlls


def main(sigma, samples, test_x):
    
    #samples = numpy.load(filename)
    
    parzen = theano_parzen(samples, sigma)
    
    test_ll = get_nll(test_x, parzen)
    
    #print "Log-Likelihood of valid set = %.5f" % valid_ll
    print "Mean Log-Likelihood of test set = %.5f" % numpy.mean(test_ll)
    print "Std of Log-Likelihood of test set = %.5f" % numpy.std(test_ll)


if __name__ == "__main__":
    """
    on marge run the following command:
    jobdispatch --local=10 --file=parzen_likelihood_commands

    0.2 is a good value for TFD
    """
    main(float(sys.argv[1]), sys.argv[2])
    
