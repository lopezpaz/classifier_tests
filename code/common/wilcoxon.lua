local distributions = require 'distributions'

local function wilcox_test(x,y)
  if(x:size(2) > 1) then
    return nil, nil
  end
  local nx = x:size(1)
  local ny = y:size(1)
  local d  = torch.cat(x:view(-1),y:view(-1))

  local _, id = torch.sort(d:view(-1))
  local _, rd = torch.sort(id)

  local ux = nx*ny + (nx*(nx+1))/2.0 - rd[{{1,nx}}]:sum()
  local uy = nx*ny - ux

  local m = nx*ny/2.0 + 0.5
  local s = math.sqrt((nx*ny*(nx+ny+1.0))/12.0)
  local t = (uy-m)/s

  return t, 1.0-distributions.norm.cdf(t,0,1)
end

return wilcox_test
