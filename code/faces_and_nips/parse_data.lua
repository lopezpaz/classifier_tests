local pl_data = require 'pl.data'

os.execute('parse_data.py')

local files = {
  'bayes_bayes',
  'bayes_deep',
  'bayes_learning',
  'bayes_neuro',
  'deep_learning',
  'neuro_learning',
  'faces_diff',
  'faces_same'
}

for f=1,#files do
  print(files[f])
  local p = torch.Tensor(pl_data.read(files[f]..'_p.txt'))
  local q = torch.Tensor(pl_data.read(files[f]..'_q.txt'))
  torch.save(files[f]..'.t7', { p=p, q=q })
end

os.execute('rm *.txt')
