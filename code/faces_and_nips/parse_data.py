import numpy as np
import sys
sys.path.append("../../../") # https://github.com/wittawatj/interpretable-test
import freqopttest.data as data

documents = [
  "bayes_bayes_np430_nq432_d2000.p",
  "bayes_deep_np846_nq433_d2000_random_noun.p",
  "bayes_learning_np821_nq276_d2000.p",
  "bayes_neuro_np794_nq788_d2000_random_noun.p",
  "deep_neuro_np105_nq512_d2000.p",
  "deep_learning_np431_nq299_d2000_random_noun.p",
  "neuro_learning_np832_nq293_d2000.p"
]

for i in xrange(len(documents)):
  print(documents[i])
  x  = np.load('../'+documents[i])
  f  = documents[i].split('_')
  p  = x['P']
  q  = x['Q']
  fp = f[0]+'_'+f[1]+'_'+'p.txt'
  fq = f[0]+'_'+f[1]+'_'+'q.txt'
  np.savetxt(fp,p)
  np.savetxt(fq,q)

faces = np.load('../crop48_HANESU_AFANDI.p')
np.savetxt('faces_diff_p.txt', faces.X)
np.savetxt('faces_diff_q.txt', faces.Y)

faces = np.load('../crop48_h0.p')
np.savetxt('faces_same_p.txt', faces)
np.savetxt('faces_same_q.txt', faces)
