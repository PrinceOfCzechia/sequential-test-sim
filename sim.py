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
p0 = 0.30
p1 = 0.70

def L1( p, k, n ):
    '''
    probability of accepting H0 if the true value of the parameter is p
    '''
    return sum( choose( j, n ) * p**j * ( 1-p )**( n-j ) for j in range( int(k+1) ) )

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

'''
H = test1( eval_k( n, p0, alpha ), verbose = True )
print( H )
'''

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
arr_L = L1( arr_p, eval_k( n, p, alpha ), n )

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

p = 0.20
p0 = 0.10
p1 = 0.40

def eval_a( n, p0, alpha ):
    return binom.ppf( 1-alpha/2, n, p0 )

def eval_b( n, p1, alpha ):
    return binom.ppf( alpha/2, n, p1 )

def L2( p, a, b, n1, n2 ):
    if( a > n1 ):
        print('Error in the formula, b-j < 0')
        return 2
    else: return L1( p, a, n1 ) + sum( choose( j, n1 ) * p**j * ( 1-p )**( n1-j ) * L1( p, b-j, n2 ) for j in range( int(a+1), int(b+1) ) )

def EN( p, n1, n2, alpha ):
    a = eval_a( n1, p0, alpha )
    b = eval_b( n1, p1, alpha )
    # print( round( L1( p, b, n1 ), 4 ), round( L1( p, a, n1 ), 4 ) )
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
        b = eval_b( n1+n2, p1, alpha ) # adjust b
        Sn = np.sum( x ) # recalculate test statistic
        if verbose: print( 'b =', b, ', Sn =', Sn)
        if( Sn <= b ): return False
        else: return True

'''
print( test2( n1, n2, alpha, p0, p1, verbose = True ) )
print( 'EN =', round( EN( p, n1, n2, alpha ), ndigits = 4) )
'''

arr_p = np.linspace( 0, 1, 201 )
arr_EN = EN( arr_p, n1, n2, alpha )

'''
# plot EN for different p
plt.plot( arr_p, arr_EN )
plt.xlabel( 'p' )
plt.ylabel( 'EN(p)' )
plt.show()
'''

def test2_count( n1, n2, alpha, p0, p1 ):
    '''
    return 0 if decided without sample extension
    return 1 if sample extended
    '''
    a = eval_a( n1, p0, alpha )
    b = eval_b( n2, p1, alpha )
    x = np.array( [ gen_X( p ) for i in range( n1 ) ] ) # draw the sample
    Sn = np.sum( x ) # test statistic
    if( Sn <= a ): return 0
    elif( Sn > b ): return 0
    else: return 1

'''
TODO: revisit
mean_E = 0
for i in range(5000):
    mean_E += n1 + n2 * test2_count( n1, n2, alpha, p0, p1 )

mean_E /= 5000
print('Empirical mean N =', mean_E, 'versus theoretical EN =', EN( p, n1, n2, alpha ) )
'''

'''
arr_L = L2( arr_p, eval_a( n, p0, alpha ), eval_b( n, p1, alpha), n1, n2 )
print( 'p0 =', p0, 'p1 =', p1 )
plt.plot( arr_p, arr_L )
plt.axvline( x=p0, color='red', linestyle='--' )
plt.axvline( x=p1, color='green', linestyle='--' )
plt.xlabel( 'p' )
plt.ylabel( 'L2(p)' )
plt.show()
'''

'''
3) Curtailed-sampling test
'''

N = 200 # maximum sample size
n = 20 # single batch size

p0 = 0.3
p1 = 0.5
alpha = 0.05

p = 0.3

def eval_c( N, p0, alpha ): # TODO: check
    return binom.ppf( 1-alpha, N, p0 )

def test3( n, N, p0, alpha ):
    c = eval_c( N, p0, alpha )
    x = np.array( [ gen_X( p ) for i in range( n ) ] ) # draw the sample
    Sn = np.sum( x ) # test statistic
    while n < N and Sn <= c:
        if( Sn == c ): return True
        else:
            x = np.append( x, gen_X( p ) )
            Sn = sum( x )
            n += 1
    return False

print( test3( n, N, p0, alpha ) )

success_counter = 0
for i in range(1000):
    success_counter += test3( n, N, p0, alpha )

print( 'Empirical test level', success_counter/1000 )

'''
4) Wald sequential test
'''

n = 20

alpha = 0.05
beta = 0.05

p0 = 0.45
p1 = 0.55

a = math.log( (1-beta) / alpha )
b = math.log( beta / (1-alpha) )

ha = a / math.log( (p1*(1-p0)) / (p0*(1-p1)) )
hb = b / math.log( (p1*(1-p0)) / (p0*(1-p1)) )

s = math.log( (1-p0)/(1-p1) ) / math.log( (p1*(1-p0)) / (p0*(1-p1)) )

# Qn = ( (p1*(1-p0)) / (p0*(1-p0)) )**(sum( x )) * ( (1-p1) / (1-p0) )**n

def test4( p, n, verbose = False ): # (n*counter) is the sample size after extension
    counter = 1
    x = np.array( [ gen_X( p ) for i in range( n ) ] )
    while hb + (n*counter)*s < sum( x ) and sum( x ) < ha + (n*counter)*s:
        counter += 1
        x = np.append( x, np.array( [ gen_X( p ) for i in range( n ) ] ) ) # extend the sample
    if verbose:
        print( 'data = \n', np.where( x, 1, 0 ).reshape( -1, n ) )
        print( counter-1, 'sample extensions required' )
    if sum( x ) < hb + (n*counter)*s: return False
    else: return True


'''
print( test4( 0.46, 5, verbose = True ) )
'''

'''
5) Curtailed Wald test
'''

def test5( p, n, N, verbose = False ): # (n*counter) is the sample size after extension
    counter = 1
    x = np.array( [ gen_X( p ) for i in range( n ) ] )
    while hb + (n*counter)*s < sum( x ) and sum( x ) < ha + (n*counter)*s:
        counter += 1
        x = np.append( x, np.array( [ gen_X( p ) for i in range( n ) ] ) ) # extend the sample
        if n*counter >= N: break
        else: pass
    if verbose:
        print( 'data = \n', np.where( x, 1, 0 ).reshape( -1, n ) )
        print( counter-1, 'sample extensions required' )
        if( n*counter >= N ): print( 'Stopped at limit N =', N )
        else: print( 'Stopped naturally' )
    if( n*counter >= N ): # limit reached
        if sum( x ) < len( x ) * s: return False
        else: return True
    elif sum( x ) < hb + (n*counter)*s: return False # rule from test4
    else: return True

'''
print( test5( 0.46, n = 20, N = 100, verbose = True ) )
'''


'''
6.1) Normal distribution variance test, known expectation
'''

alpha = 0.05
beta = 0.05

mu = 0
sigma = 4

sigma0 = 1
sigma1 = 4

a = math.log( (1-beta) / alpha )
b = math.log( beta / (1-alpha) )

ha = 2*a / ( sigma0**(-2) - sigma1**(-2) )
hb = 2*b / ( sigma0**(-2) - sigma1**(-2) )
s = 2*math.log( sigma1/sigma0) / ( sigma0**(-2) - sigma1**(-2) )

n = 2

def testN1( n, ha, hb, s, mu, sigma, verbose = False ):
    counter = 1
    x = np.random.normal( mu, math.sqrt(sigma), size = n )
    while hb + (n*counter)*s < sum( (x-mu)**2 ) and sum( (x-mu)**2 ) < ha + (n*counter)*s:
        counter += 1
        x = np.append( x, np.random.normal( mu, math.sqrt(sigma), size = n ) ) # extend the sample
    if verbose:
        print( 'data = \n', np.round( x, decimals = 3 ).reshape( -1, n ) )
        print( counter-1, 'sample extensions required' )
    if sum( (x-mu)**2 ) < hb + (n*counter)*s: return False
    else: return True

'''
print( testN1( n, ha, hb, s, mu, sigma, verbose = True ) )
'''


'''
6.2) Normal distribution variance test, unknown expectation
'''

alpha = 0.05
beta = 0.05

mu = 0
sigma = 4

sigma0 = 1
sigma1 = 4

a = math.log( (1-beta) / alpha )
b = math.log( beta / (1-alpha) )

ha = 2*a / ( sigma0**(-2) - sigma1**(-2) )
hb = 2*b / ( sigma0**(-2) - sigma1**(-2) )
s = 2*math.log( sigma1/sigma0) / ( sigma0**(-2) - sigma1**(-2) )

def testN2( n, ha, hb, s, mu, sigma, verbose = False ):
    counter = 1
    x = np.random.normal( mu, math.sqrt(sigma), size = n )
    while hb + (n*counter)*s < sum( ( x - np.mean(x) )**2 ) and sum( ( x - np.mean(x) )**2 ) < ha + (n*counter)*s:
        counter += 1
        x = np.append( x, np.random.normal( mu, math.sqrt(sigma), size = n ) ) # extend the sample
    if verbose:
        print( 'data = \n', np.round( x, decimals = 3 ).reshape( -1, n ) )
        print( counter-1, 'sample extensions required' )
    if sum( ( x - np.mean(x) )**2 ) < hb + (n*counter)*s: return False
    else: return True

'''
mu = np.random.uniform( 0.0, 100.0, 1) # so that mu is truly unknown
print( testN1( n, ha, hb, s, mu, sigma, verbose = True ) )
'''