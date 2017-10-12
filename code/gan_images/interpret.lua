local metal = require 'metal'
local image = require 'image'
local optim = require 'optim'

function train_test_split(x,y,p_tr)
  local p_tr = p_tr or 0.5
  local n_tr = math.floor(x:size(1)*p_tr)
  local i = torch.randperm(x:size(1)):long()
  local i_tr = i[{{1,n_tr}}]
  local i_te = i[{{n_tr+1,x:size(1)}}]
  local x_tr = x:index(1,i_tr)
  local y_tr = y:index(1,i_tr)
  local x_te = x:index(1,i_te)
  local y_te = y:index(1,i_te)
  return x_tr, y_tr, x_te, y_te
end

local function distances(x,y)
   local result = x.new(x:size(1),y:size(1)):fill(0)
   result:addmm(x,y:t()):mul(-2)
   result:add(x:clone():pow(2):sum(2):expandAs(result))
   result:add(y:clone():pow(2):sum(2):t():expandAs(result))
   return result
end

local function make_dataset(p,q)
  local x = torch.cat(p,q,1)
  local y = torch.cat(torch.zeros(p:size(1),1), torch.ones(q:size(1),1), 1)
  local n = x:size(1)
  local perm = torch.randperm(n):long()
  x = x:index(1,perm):double()
  y = y:index(1,perm):double()
  return x, y 
end

local function test(p,q,q_te)
  local x, y = make_dataset(p,q)
  local net = nn.Sequential()
 
  net:add(nn.Dropout(0.5))
  net:add(nn.Linear(x:size(2),1))
  net:add(nn.Sigmoid())

  local ell = nn.BCECriterion()

  local params = { optimizer = optim.sgd }

  for i=1,20 do
    metal.train(net, ell, x, y, params)
    xlua.progress(i,20)
  end
  return metal.predict(net,q_te)
end

local dataset = 'bedrooms'
local d1 = torch.load('little_'..dataset..'_pixels.t7')
local x1 = d1[1]
local y1 = d1[2]

local d2 = torch.load('little_'..dataset..'_resnet.t7')
local x2 = d2[1]
local y2 = d2[2]

-- training real pixels, training real resnet
local x1_tr, x2_tr, x1_te, x2_te = train_test_split(x1,x2,0.5)
-- training fake pixels, training fake resnet
local y1_tr, y2_tr, y1_te, y2_te = train_test_split(y1,y2,0.5)

local p = test(x2_tr, y2_tr, y2_te)
local sort_a, sort_b = torch.sort(p,1)
local selected_i = torch.linspace(1,sort_b:size(1), 20):long()
local selected = sort_b:index(1,selected_i):long():view(-1)
local n = sort_b:size(1)
print(p:min())
print(sort_a:index(1,selected_i))
local res = y1_te:index(1, selected):view(selected:size(1), 3, 64, 64)
local ttt = image.toDisplayTensor{input=res, nrow=5}
image.save('test.jpg', ttt)
