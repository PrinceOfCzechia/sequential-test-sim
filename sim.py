import numpy as np
import scipy.stats as stats
from scipy.stats import binom
from scipy.stats import bernoulli # TODO: rethink and remove

'''
assume that
X1, X2, ... be iid random variables with Xi ~ Alt(p), i.e.
    P(Xi = 1) = p
    P(Xi = 0) = 1 - p
we aim to test the hypotheses
H0: p = p0
H1: p = p1
0 < p0 < p1 < 1
'''


def gen_X( p ):
    '''
    draw Xi from the pre-specified distribution
    with the parameter value set to 'p'
    '''
    r = np.random.uniform( 0.0, 1.0, 1 )
    return( r < p )


'''
1) Test with a fixed sample size
'''
n = 100
alpha = 0.05
p = 0.30
p0 = 0.30
p1 = 0.70
x = np.empty( n )


def eval_k( n, p0, alpha ):
    return binom.ppf( 1-alpha, n, p0 )

def test1( k ):
    x = np.array( [ gen_X( p ) for i in range( n ) ] )
    Sn = np.sum( x )

    # print( 'Sn = ', Sn )
    # print( 'k = ', k)

    return( Sn > k)

lvl = 0
for i in range( 1000 ):
    H = test1( eval_k( n, p0, alpha ) )
    if( H ): lvl += 1

print( lvl/1000 )


'''
2) Two-stage test
'''
n1 = 50
n2 = 50


'''
3) 
'''


'''
4) 
'''


'''
5) 
'''