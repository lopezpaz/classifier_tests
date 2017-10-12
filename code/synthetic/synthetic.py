import argparse
import numpy as np

from freqopttest.data import TSTData
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.tst as tst
import freqopttest.kernel as kernel
import freqopttest.glo as glo

import sklearn
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

if (sklearn.__version__ == '0.19.dev0'):
  from sklearn.model_selection import GridSearchCV
  from sklearn.model_selection import train_test_split
else:
  from sklearn.grid_search import GridSearchCV
  from sklearn.cross_validation import train_test_split

from scipy.stats import ks_2samp, ranksums
from scipy.stats import norm, binom

from kuiper import kuiper_two

parser = argparse.ArgumentParser(description='Synthetic data experiment')
parser.add_argument('--reps', default='1')
parser.add_argument('--test', default='2')
args = parser.parse_args()
            
def wtest(p,q,alpha=0.05):
    op = {'n_test_locs': 2,
          'seed': 0,
          'max_iter': 200, 
          'batch_proportion': 1.0,
          'locs_step_size': 1.0, 
          'gwidth_step_size': 0.1,
          'tol_fun': 1e-4
         }
    if (p.ndim == 1): p = p[:,np.newaxis]
    if (q.ndim == 1): q = q[:,np.newaxis]
    d = data.TSTData(p,q)
    d_tr, d_te = d.split_tr_te(tr_proportion=0.5)
    test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(d_tr, alpha, **op)
    met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
    r = met_opt.perform_test(d_te)
    if (r['test_stat'] == -1):
      r['test_stat'] = np.nan
      r['pvalue'] = np.nan
    return r['test_stat'], r['pvalue']

def mmd(p,q,alpha=0.05):
    if (p.ndim == 1): p = p[:,np.newaxis]
    if (q.ndim == 1): q = q[:,np.newaxis]
    d = data.TSTData(p,q)
    d_tr, d_te = d.split_tr_te(tr_proportion=0.5)
    med = util.meddistance(d_tr.stack_xy())
    widths = [ (med*f) for f in 2.0**np.linspace(-1, 4, 20)]
    list_kernels = [kernel.KGauss( w**2 ) for w in widths]
    besti, powers = tst.LinearMMDTest.grid_search_kernel(d_tr, list_kernels, alpha)
    best_ker = list_kernels[besti]
    lin_mmd_test = tst.LinearMMDTest(best_ker, alpha)
    r = lin_mmd_test.perform_test(d_te)
    return r['test_stat'], r['pvalue']

tests = {
    2: { 'name': 'MMD', 'foo': mmd, 'maxdim': 100 },
    3: { 'name': 'Wilcoxon', 'foo': ranksums, 'maxdim': 1 },
    4: { 'name': 'Kolmogorov-Smirnov', 'foo': ks_2samp, 'maxdim': 1 },
    5: { 'name': 'Kuiper', 'foo': kuiper_two, 'maxdim': 1},
    6: { 'name': 'ME', 'foo': wtest, 'maxdim': 100}
}

n_grid     = [50, 100, 500, 1000, 2000]
df_grid    = [1, 2, 5, 10, 15, 20]
f_grid     = [2, 4, 6, 8, 10, 20]
v_grid     = [0.1, 0.25, 0.5, 1, 2, 3]
n_default  = 2000
f_default  = 1
v_default  = 0.25
df_default = 3
reps       = 100
alpha      = 0.05

def gen_student(n,df):
    p = scale(np.random.randn(n))
    q = scale(np.random.standard_t(df,(n)))
    return p,q

def gen_sinusoid(n,f,v):
    x = scale(np.random.randn(n,1))
    y = scale(np.cos(x*f)+np.random.randn(n,1)*v)
    p = np.random.permutation(n)
    return np.concatenate((x,y),1), np.concatenate((x,y[p]),1)

for testi in tests.keys():
  test = tests[testi]['foo']
  for n in n_grid:
      for rep in xrange(reps):
          p, q = np.random.randn(n), np.random.randn(n)
          t = test(p,q)
          print('%d type1_n %d 1 -1 %f %f %d' % (testi, n, t[0], t[1], t[1] > alpha))
  
  for n in n_grid:
      for rep in xrange(reps):
          p, q = gen_student(n, df_default)
          t = test(p,q)
          print('%d student_n %d 1 %d %f %f %d' % (testi, n, df_default, t[0], t[1], t[1] <= alpha))
 
  for df in df_grid:
      for rep in xrange(reps):
          p, q = gen_student(n_default, df)
          t = test(p,q)
          print('%d student_df %d 1 %d %f %f %d' % (testi, n_default, df, t[0], t[1], t[1] <= alpha))
 
  if (tests[testi]['maxdim'] > 1):
      for n in n_grid:
          for rep in xrange(reps):
              p, q = gen_sinusoid(n,f_default,v_default)
              t = test(p,q)
              print('%d sinusoid_n %d %d %f %f %f %d' % (testi, n, f_default, v_default, t[0], t[1], t[1] <= alpha))
  
      for f in f_grid:
          for rep in xrange(reps):
              p, q = gen_sinusoid(n_default,f,v_default)
              t = test(p,q)
              print('%d sinusoid_f %d %f %f %f %f %d' % (testi, n_default, f, v_default, t[0], t[1], t[1] <= alpha))
  
      for v in v_grid:
          for rep in xrange(reps):
              p, q = gen_sinusoid(n_default,f_default,v)
              t = test(p,q)
              print('%d sinusoid_v %f %f %f %f %f %d' % (testi, n_default, f_default, v, t[0], t[1], t[1] <= alpha))
