import pickle
import numpy as np
from subprocess import call

call(["wget", "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip"])
call(["unzip", "-d", "pairs", "pairs.zip"])
call(["rm", "pairs.zip"])

meta = np.genfromtxt('pairs/pairmeta.txt', dtype=np.str)

weights = []
samples = []
labels  = []

for i in xrange(meta.shape[0]):
  d = np.genfromtxt('pairs/pair' + meta[i][0] + '.txt')
  x = d[:,0]
  y = d[:,1]

  if((meta[i][1] == '1') and
     (meta[i][2] == '1') and
     (meta[i][3] == '2') and
     (meta[i][4] == '2')):
    samples.append(np.vstack((x,y)).T)
    labels.append(0)
    weights.append(float(meta[i][5]))
    samples.append(np.vstack((y,x)).T)
    labels.append(1)
    weights.append(float(meta[i][5]))
  
  if((meta[i][1] == '2') and
     (meta[i][2] == '2') and
     (meta[i][3] == '1') and
     (meta[i][4] == '1')):
    samples.append(np.vstack((y,x)).T)
    labels.append(0)
    weights.append(float(meta[i][5]))
    samples.append(np.vstack((x,y)).T)
    labels.append(1)
    weights.append(float(meta[i][5]))

f = open('tuebingen.pkl', 'w')
pickle.dump((samples, labels, weights), f)
f.close()

call(["rm", "-rf", "pairs"])
