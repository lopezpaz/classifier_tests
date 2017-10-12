local metal = require 'metal'
local distributions = require 'distributions'
local optim = require 'optim'

local function make_dataset(p,q)
  local x = torch.cat(p,q,1)
  local y = torch.cat(torch.zeros(p:size(1),1), torch.ones(q:size(1),1), 1)
  local n = x:size(1)
  return metal.train_test_split(x,y)
end

local function neural_test(p,q)
  local x_tr, y_tr, x_te, y_te = make_dataset(p,q)
  local epochs = epochs or 100
  local h = h or 20

  local net = nn.Sequential()
  net:add(nn.Linear(x_tr:size(2),h))
  net:add(nn.ReLU())
  net:add(nn.Linear(h,1))
  net:add(nn.Sigmoid())

  local ell = nn.BCECriterion()

  local params = { optimizer = optim.adam }

  for i=1,epochs do
    metal.train(net, ell, x_tr, y_tr, params)
  end

  local loss, acc = metal.eval(net, ell, x_te, y_te, params)
  local cdf = distributions.norm.cdf(acc,0.5,torch.sqrt(0.25/x_te:size(1)))
  return acc, 1.0-cdf
end

local function experiment(dir,name,seed)
  local d = torch.load(dir..name..'.t7')
  local p = d.p
  local q = d.q

  if(name == 'faces_same') then
    local i = torch.randperm(p:size(1))
    p = p:index(1,i:long())
    q = p[{{p:size(1)/2+1,p:size(1)},{}}]
    p = p[{{1,p:size(1)/2},{}}]
  end
    
  if(p:size(1) > q:size(1)) then
    local i = torch.randperm(p:size(1))[{{1,q:size(1)}}]
    p = p:index(1,i:long())
  elseif(q:size(1) > p:size(1)) then
    local i = torch.randperm(q:size(1))[{{1,p:size(1)}}]
    q = q:index(1,i:long())
  end

  print(seed, name, q:size(1), p:size(1), neural_test(p,q))
end 

local dir = '.' -- https://github.com/wittawatj/interpretable-test

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

local cmd = torch.CmdLine()
cmd:option('--seed',1)
local params = cmd:parse(arg)

for f=1,#files do
  experiment(dir,files[f],params.seed)
end
