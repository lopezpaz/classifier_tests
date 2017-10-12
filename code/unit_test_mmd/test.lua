require 'cephes'
data = require 'pl.data'

local function rbf(x,y,g)
  local d = -1.0/(2*(g^2))
  return torch.exp(torch.mul(torch.sum(torch.pow(x-y,2),2),d))
end

local function mmd_stat(x,y,g)
  local m   = x:size(1)
  local m2  = torch.ceil(m/2.0)
  local kxx = rbf(x[{{1,m2}}], x[{{m2+1,m}}],g)
  local kyy = rbf(y[{{1,m2}}], y[{{m2+1,m}}],g)
  local kxy = rbf(x[{{1,m2}}], y[{{m2+1,m}}],g)
  local kyx = rbf(x[{{m2+1,m}}], y[{{1,m2}}],g)
  local res = kxx+kyy-kxy-kyx
  return res:mean(), res:var()/m2
end

local function choose_g(x,y)
  local g  = torch.pow(2,torch.range(-15,10))
  local l  = 10e-4
  local m2 = torch.ceil(x:size(1)/2.0)
  local bg = 1
  local br = 0

  for i=1,g:size(1) do
    local mean, variance = mmd_stat(x,y,g[i])
    local ratio = mean/(torch.sqrt(variance*m2)+l)
    if (ratio > br) then
      br = ratio
      bg = g[i]
    end
  end
  return bg
end

local function mmd_test(x, y, g, alpha)
  local a  = alpha or 0.05

  local g  = g or choose_g(x,y)
  local mmd_m, mmd_v = mmd_stat(x,y,g)

  local cdf = 0.5*(1.0+cephes.erf((mmd_m)/math.sqrt(2*mmd_v)))
  return g, mmd_m, mmd_v, 1-cdf 
end

local p = torch.Tensor(data.read('p.txt'))
local q = torch.Tensor(data.read('q.txt'))

print('matlab',4.0000e+00,1.4896e-03, 2.2005e-06,1.5764e-01)
print('torch', mmd_test(p,q))

