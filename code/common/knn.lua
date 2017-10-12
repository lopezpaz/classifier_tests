local distributions = require 'distributions'

local function distances(x,y)
   local result = x.new(x:size(1),y:size(1)):fill(0)
   result:addmm(x,y:t()):mul(-2)
   result:add(x:clone():pow(2):sum(2):expandAs(result))
   result:add(y:clone():pow(2):sum(2):t():expandAs(result))
   return result
end

local function make_dataset(p,q,p_tr)
  local p_tr = p_tr or 0.5
  local x = torch.cat(p,q,1)
  local y = torch.cat(torch.zeros(p:size(1),1), torch.ones(q:size(1),1), 1)
  local n = x:size(1)
  local perm = torch.randperm(n):long()
  x = x:index(1,perm)
  y = y:index(1,perm)

  local x_tr = x[{{1,torch.floor(n*p_tr)}}]:clone()
  local y_tr = y[{{1,torch.floor(n*p_tr)}}]:clone()
  local x_te = x[{{torch.floor(n*p_tr)+1,n}}]:clone()
  local y_te = y[{{torch.floor(n*p_tr)+1,n}}]:clone()

  return x_tr, y_tr, x_te, y_te
end

local function knn_test(p,q, params)
  local params = params or { ptr = 0.5 }
  local x_tr, y_tr, x_te, y_te = make_dataset(p,q, params.ptr)
  local p_te = torch.zeros(x_te:size(1))
  local k = k or math.sqrt(x_tr:size(1))
  local t = torch.ceil(k/2)

  local sort_a, sort_b = torch.sort(distances(x_tr,x_te),1)
  sort_b = sort_b[{{1,k}}]:t()

  for i=1,x_te:size(1) do
    if (y_tr:index(1,sort_b[i]):sum() > t) then
      p_te[i] = 1
    end
  end

  local t = torch.eq(p_te,y_te):float():mean()
  local cdf = distributions.norm.cdf(t,0.5,torch.sqrt(0.25/x_te:size(1)))
  return t, 1.0-cdf
end

return knn_test
