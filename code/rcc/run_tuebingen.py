def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from sklearn.ensemble      import RandomForestClassifier as RFC
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale 
from sklearn.mixture       import GMM 
from sklearn.linear_model  import LogisticRegression as LR
import pickle
from tqdm import tqdm

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T
def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))
def f2(x,y,z):
  return np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(z,wz).mean(0)))

def cause(n,k,p1,p2):
  g = GMM(k)
  g.means_   = p1*np.random.randn(k,1)
  g.covars_  = np.power(abs(p2*np.random.randn(k,1)+1),2)
  g.weights_ = abs(np.random.rand(k,1))
  g.weights_ = g.weights_/sum(g.weights_)
  return scale(g.sample(n))

def noise(n,v):
  return v*np.random.rand(1)*np.random.randn(n,1)

def mechanism(x,d):
  g = np.linspace(min(x)-np.std(x),max(x)+np.std(x),d);
  return sp(g,np.random.randn(d))(x.flatten())[:,np.newaxis]

def pair(n=1000,k=3,p1=2,p2=2,v=2,d=5):
  x  = cause(n,k,p1,p2)
  return (x,scale(scale(mechanism(x,d))+noise(n,v)))

def pairset(N):
  z1 = np.zeros((N,3*wx.shape[1]))
  z2 = np.zeros((N,3*wx.shape[1]))
  for i in tqdm(range(N)):
    (x,y)   = pair()
    z1[i,:] = f2(x,y,np.hstack((x,y)))
    z2[i,:] = f2(y,x,np.hstack((y,x)))
  return (np.vstack((z1,z2)),np.hstack((np.zeros(N),np.ones(N))).ravel(),np.ones(2*N))

def tuebingen(fname = 'tuebingen.pkl'):
  f = open(fname, 'r')
  x,y,w = pickle.load(f)
  f.close()

  print(len(x))
  
  z = np.zeros((len(x),3*wx.shape[1]))
  for i in xrange(len(x)):
    a = scale(x[i][:,0])[:,np.newaxis]
    b = scale(x[i][:,1])[:,np.newaxis]
    z[i,:] = f2(a,b,np.hstack((a,b)))
  
  return z,y,w

np.random.seed(0)

N = 10000
K = 333
E = 500

# (1.7,1.9,1.9), (1.5)
wx = rp(K,[0.15,1.5,15],1)
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.15,1.5,15],2)

(x1,y1,m1) = pairset(N)
(x2,y2,m2) = pairset(N)
(x0,y0,m0) = tuebingen()

reg  = RFC(n_estimators=E,random_state=0,n_jobs=16).fit(x1,y1);
print [N,K,E,reg.score(x1,y1,m1),reg.score(x2,y2,m2),reg.score(x0,y0,m0)]
