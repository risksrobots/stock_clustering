"""
Sample code automatically generated on 2022-04-27 16:14:24

by www.matrixcalculus.org

from input

d/dweight ((weight'*returns)-(riskless)')/(weight'*covmatr*weight).^0.5 = 1/(weight'*covmatr*weight).^0.5*returns-((0.5*(weight'*covmatr*weight).^(-0.5)*(weight'*returns-riskless))/((weight'*covmatr*weight).^0.5).^2*covmatr*weight+(0.5*(weight'*covmatr*weight).^(-0.5)*(returns'*weight-riskless))/((weight'*covmatr*weight).^0.5).^2*covmatr*weight)

where

covmatr is a symmetric matrix
returns is a vector
riskless is a scalar
weight is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(covmatr, returns, riskless, weight):
    assert isinstance(covmatr, np.ndarray)
    dim = covmatr.shape
    assert len(dim) == 2
    covmatr_rows = dim[0]
    covmatr_cols = dim[1]

    dim = returns.shape
    assert len(dim) == 1
    returns_rows = dim[0]
    if isinstance(riskless, np.ndarray):
        dim = riskless.shape
        assert dim == (1, )
    assert isinstance(weight, np.ndarray)
    dim = weight.shape
    assert len(dim) == 1
    weight_rows = dim[0]
    assert covmatr_rows == covmatr_cols == returns_rows == weight_rows

    t_0 = (covmatr).dot(weight)
    t_1 = (weight).dot(t_0)
    t_2 = (t_1 ** 0.5)
    t_3 = ((weight).dot(returns) - riskless)
    t_4 = (0.5 * (t_1 ** -0.5))
    t_5 = (t_2 ** 2)
    functionValue = (t_3 / t_2)
    gradient = (((1 / t_2) * returns) - ((((t_4 * t_3) / t_5) * t_0) + (((t_4 * ((returns).dot(weight) - riskless)) / t_5) * t_0)))

    return functionValue, gradient

def checkGradient(covmatr, returns, riskless, weight):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(covmatr, returns, riskless, weight + t * delta)
    f2, _ = fAndG(covmatr, returns, riskless, weight - t * delta)
    f, g = fAndG(covmatr, returns, riskless, weight)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    covmatr = np.random.randn(3, 3)
    covmatr = 0.5 * (covmatr + covmatr.T)  # make it symmetric
    returns = np.random.randn(3)
    riskless = np.random.randn(1)
    weight = np.random.randn(3)

    return covmatr, returns, riskless, weight

if __name__ == '__main__':
    covmatr, returns, riskless, weight = generateRandomData()
    functionValue, gradient = fAndG(covmatr, returns, riskless, weight)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(covmatr, returns, riskless, weight)
