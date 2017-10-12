require 'distributions'

local function rbf(x,y,g)
  local r = torch.add(x,y)
  local d = -1.0/(2*(g^2))
  return torch.exp(torch.mul(torch.sum(torch.pow(x-y,2),2),d))
end

local function mmd_stat(x,y,g)
  local m   = x:size(1)
  if ((m%2)==1) then
    m = m-1
  end
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

local function mmd_test(x, y, use_all)
  local m  = x:size(1)
  local m2 = torch.ceil(m/2.0)
  local g  = g or choose_g(x[{{1,m2}}],y[{{1,m2}}])
  mmd_m, mmd_v = mmd_stat(x[{{m2+1,m}}],y[{{m2+1,m}}],g)

  local cdf = distributions.norm.cdf(mmd_m,0,math.sqrt(mmd_v))
  return mmd_m, 1.0-cdf
end

return mmd_test
