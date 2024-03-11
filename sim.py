import numpy as np

'''
dal bude platit
X1, X2, ... nahodne veliciny splnujici
    P(Xi = 1) = p
    P(Xi = 0) = 1 - p
H0: p = p0
H1: p = p1
0 < p0 < p1 < 1
'''

def gen_X( p ):
    return( np.random.uniform( 0.0, 1.0, 1 ) )

'''
1) Test s pevnym rozsahem vyberu
'''
n = 100
alpha = 0.05
x = np.random.uniform( 0.0, 1.0, n )

Sn = np.sum( x )
print( Sn )