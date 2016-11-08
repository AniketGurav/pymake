import math
import numpy as np
import scipy as sp

##########################
### Stochastic Process
##########################

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(np.random.multinomial(1, params) == 1)[0]

def bernoulli(param, size=1):
    return np.random.binomial(1, param, size=size)

### Power law distribution generator
def random_powerlaw(alpha, x_min, size=1):
    ### Discrete
    alpha = float(alpha)
    u = np.random.random(size)
    x = (x_min-0.5)*(1-u)**(-1/(alpha-1))+0.5
    return np.floor(x)

### A stick breakink process, truncated at K components.
def gem(gmma, K):
    sb = np.empty(K)
    cut = np.random.beta(1, gmma, size=K)
    for k in range(K):
        sb[k] = cut[k] * cut[0:k].prod()
    return sb


##########################
### Means and Norms
##########################

### Weighted  means
def wmean(a, w, mean='geometric'):
    if mean == 'geometric':
        kernel = lambda x : np.log(x)
        out = lambda x : np.exp(x)
    elif mean == 'arithmetic':
        kernel = lambda x : x
        out = lambda x : x
    elif mean == 'harmonic':
        num = np.sum(w)
        denom = np.sum(np.asarray(w) / np.asarray(a))
        return num / denom
    else:
        raise NotImplementedError('Mean Unknwow: %s' % mean)

    num = np.sum(np.asarray(w) * kernel(np.asarray(a)))
    denom = np.sum(np.asarray(w))
    return out(num / denom)

##########################
### Matrix Operation
##########################

def draw_square(mat, value, topleft, l, L, w=0):
    tl = topleft

    # Vertical draw
    mat[tl[0]:tl[0]+l, tl[1]:tl[1]+w] = value
    mat[tl[0]:tl[0]+l, tl[1]+L-w:tl[1]+L] = value
    # Horizontal draw
    mat[tl[0]:tl[0]+w, tl[1]:tl[1]+L] = value
    mat[tl[0]+l-w:tl[0]+l, tl[1]:tl[1]+L] = value
    return mat

##########################
### Colors Operation
##########################

def floatRgb(mag, cmin, cmax):
	""" Return a tuple of floats between 0 and 1 for the red, green and
		blue amplitudes.
	"""

	try:
		# normalize to [0,1]
		x = float(mag-cmin)/float(cmax-cmin)
	except:
		# cmax = cmin
		x = 0.5
	blue = min((max((4*(0.75-x), 0.)), 1.))
	red  = min((max((4*(x-0.25), 0.)), 1.))
	green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
	return (red, green, blue)

def strRgb(mag, cmin, cmax):
   """ Return a tuple of strings to be used in Tk plots.
   """

   red, green, blue = floatRgb(mag, cmin, cmax)
   return "#%02x%02x%02x" % (red*255, green*255, blue*255)

def rgb(mag, cmin, cmax):
   """ Return a tuple of integers to be used in AWT/Java plots.
   """

   red, green, blue = floatRgb(mag, cmin, cmax)
   return (int(red*255), int(green*255), int(blue*255))

def htmlRgb(mag, cmin, cmax):
   """ Return a tuple of strings to be used in HTML documents.
   """
   return "#%02x%02x%02x"%rgb(mag, cmin, cmax)

