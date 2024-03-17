import math
import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import matplotlib.pyplot as plt

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
    return math.factorial( n ) / ( math.factorial( j ) * math.factorial( n-j ) )

'''
1) Test with a fixed sample size
'''
n = 100
alpha = 0.05
p = 0.30
p0 = 0.15
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
# plot operational characteristic
arr_p = np.linspace( 0, 1, 201 )
arr_L = L1( arr_p, eval_k( n, p0, alpha ), n )

plt.plot( arr_p, arr_L )
plt.axvline( x=p0, color='red', linestyle='--' )
plt.xlabel( 'p' )
plt.ylabel( 'L1(p)' )
plt.show()
'''

'''
2) Two-stage test
'''
n1 = 50
n2 = 50

def eval_a( n, p0, alpha ):
    return binom.ppf( 1-alpha, n, p0 )

def eval_b( n, p1, alpha ):
    return binom.ppf( alpha, n, p1 )

def L2( p, a, b, n1, n2 ):
    return L1( p, a, n1 ) + sum( choose( j, n1 ) * p**j * ( 1-p )**( n1-j ) * L1( p, b-j, n2 ) for j in range( int(a+1), int(b) ) )

def EN( n1, n2, alpha ):
    a = eval_a( n1, p, alpha )
    b = eval_b( n1+n2, p, alpha )
    print( L1( p, b, n1 ), L1( p, a, n1 ) )
    return n1 + n2*( L1( p, b, n1 ) - L1( p, a, n1) )

def test2( n1, n2, alpha, p0, p1, verbose=False ):
    a = eval_a( n1, p0, alpha )
    b = eval_b( n2, p1, alpha )
    x = np.array( [ gen_X( p ) for i in range( n1 ) ] ) # draw the sample
    Sn = np.sum( x ) # test statistic
    if verbose: print( 'a =', a, ', b =', b, ', Sn =', Sn )
    if( Sn <= a ): return False
    elif( Sn > b ): return True
    else:
        x = np.append( x, np.array( [ gen_X( p ) for i in range( n2 ) ] ) ) # extend the sample
        b = eval_b( n1+n2, alpha, p1 )
        Sn = np.sum( x ) # recalculate test statistic
        if verbose: print( 'b =', b, ', Sn =', Sn)
        if( Sn <= b ): return False
        else: return True

print( test2( n1, n2, alpha, p0, p1, True ) )
print( 'EN =', EN( n1, n2, alpha ) )

'''
arr_p = np.linspace( 0, 1, 201 )
arr_L = L2( arr_p, eval_a( n, p0, alpha ), eval_b( n, p1, alpha), n1, n2 )

plt.plot( arr_p, arr_L )
plt.axvline( x=p0, color='red', linestyle='--' )
plt.xlabel( 'p' )
plt.ylabel( 'L2(p)' )
plt.show()
'''

'''
3) Curtailed-sampling test
'''

def test3():
    pass


'''
4) 
'''


'''
5) 
'''