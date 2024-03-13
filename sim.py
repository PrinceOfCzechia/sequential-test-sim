import numpy as np
import scipy.stats as stats
from scipy.stats import binom

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

def choose( j, n ):
    return np.math.factorial( n ) / ( np.math.factorial( j ) * np.math.factorial( n-j ) )

'''
1) Test with a fixed sample size
'''
n = 100
alpha = 0.05
p = 0.30
p0 = 0.30
p1 = 0.70
x = np.empty( n )

def L1( p, k, n ):
    '''
    probability of accepting H0 if the true value of the parameter is p
    '''
    return sum( choose( j, n ) * p**j * ( 1-p )**( n-j ) for j in range( int(k) ) )

def eval_k( n, p0, alpha ):
    return binom.ppf( 1-alpha, n, p0 )

def test1( k, verbose = False ):
    x = np.array( [ gen_X( p ) for i in range( n ) ] ) # draw the sample
    Sn = np.sum( x ) # test statistic
    L = L1( p, k, n )
    if verbose:
        print( 'data = \n', np.where( x, 1, 0 ).reshape( -1, 20 ) )
        print( 'Sn = ', Sn )
        print( 'k = ', k)
        print( 'L1 = ', L)
    return( Sn > k) # True - rejected H0, False - not rejected

H = test1( eval_k( n, p0, alpha ), True )
print( H )

'''
# test level
lvl = 0
for i in range( 1000 ):
    H = test1( eval_k( n, p0, alpha ) )
    if( H ): lvl += 1

print( lvl/1000 )
'''


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