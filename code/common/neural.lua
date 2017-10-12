local metal = require 'metal.lua'
local distributions = require 'distributions'
local optim = require 'optim'

local function make_dataset(p,q,p_tr)
  local p_tr = p_tr or 0.5
  local x = torch.cat(p,q,1)
  local y = torch.cat(torch.zeros(p:size(1),1), torch.ones(q:size(1),1), 1)
  local n = x:size(1)
  local perm = torch.randperm(n):long()
  x = x:index(1,perm):double()
  y = y:index(1,perm):double()

  local x_tr = x[{{1,torch.floor(n*p_tr)}}]:clone()
  local y_tr = y[{{1,torch.floor(n*p_tr)}}]:clone()
  local x_te = x[{{torch.floor(n*p_tr)+1,n}}]:clone()
  local y_te = y[{{torch.floor(n*p_tr)+1,n}}]:clone()

  return x_tr, y_tr, x_te, y_te
end

local function neural_test(p,q)
  local x_tr, y_tr, x_te, y_te = make_dataset(p,q)
  local epochs = epochs or 10000
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

return neural_test
