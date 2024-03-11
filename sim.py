import numpy as np

'''
assume that
X1, X2, ... are random variables such that
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
    '''
    r = np.random.uniform( 0.0, 1.0, 1 )
    return( r < p )


'''
1) Test with a fixed sample size
'''
n = 100
alpha = 0.05
x = np.random.uniform( 0.0, 1.0, n )

Sn = np.sum( x )
# print( Sn )