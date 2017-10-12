local gnuplot = require 'gnuplot'
local pl_data = require 'pl.data'
local optim = require 'optim'
local metal = require 'metal'

local cmd = torch.CmdLine()
cmd:option('--seed',0)
cmd:option('--dz',10)
cmd:option('--hiddens',128)
cmd:option('--epochs',100)
cmd:option('--bs',16)
cmd:option('--dr',0.25)
cmd:option('--plot',0)
cmd:option('--pairs_dir','.')
cmd:option('--lr',0.001)
cmd:option('--beta1',0.5)
cmd:option('--beta2',0.999)
cmd:option('--wd',0)
cmd:option('--remove_outliers',1)
cmd:option('--min',0)
cmd:option('--subsample',5000)
cmd:option('--reps', 1)
cmd:option('--my_test', 0)
cmd:option('--ptr', 0.5)
local params = cmd:parse(arg)
local params_str = ""
for k,v in pairs(params) do
  if (torch.type(v) == "string") then
    params_str = params_str .. "\"" .. k .. "\"" .. " : " .. "\"" .. v .. "\"" .. ", "
  else
    params_str = params_str .. "\"" .. k .. "\"" .. " : " .. v .. ", "
  end
end

torch.manualSeed(params.seed)
local test

if (params.my_test == 0) then
  test = dofile '../common/knn.lua'
end

if (params.my_test == 1) then
  test = dofile '../common/neural.lua'
end

if (params.my_test == 2) then
  test = dofile '..mmd/common/mmd.lua'
end

local function remove_outliers(x, k, a)
  local a  = a or 0.05
  local k  = k or 20
  local pk = torch.randperm(x:size(1))[{{1,k}}]
  local xk = x:index(1,pk:long()):clone()
  local d  = torch.zeros(x:size(1)):fill(1e6)

  for i=1,x:size(1) do
    for j=1,xk:size(1) do
      local dij = (x[i]-xk[j]):norm()
      if (dij < d[i]) then
        d[i] = dij
      end
    end
  end

  local top = torch.floor(x:size(1)*a)
  local d_sort, d_idx = torch.sort(d)
  local d_idx = d_idx[{{1,x:size(1)-top}}]
  return x:index(1,d_idx):clone()
end

local function read_pair(fname,params)
  local d = torch.Tensor(pl_data.read(fname))
  -- remove outliers
  if (params.remove_outliers == 1) then
    d = remove_outliers(d)
  end
  -- subsample
  if (params.subsample > 0) then
    local p = torch.randperm(d:size(1))[{{1,math.min(d:size(1),params.subsample)}}]:long()
    d = d:index(1,p)
  end
  -- normalize
  local x = metal.normalize(d[{{},{1}}])
  local y = metal.normalize(d[{{},{2}}])
  return x, y
end

local function train_conditional_gan(x,y,params)
  local Dx = x:size(2)
  local Dy = y:size(2)

  -- generator(x,z)
  local g = nn.Sequential()
  g:add(nn.Linear(Dx+params.dz,params.hiddens))
  g:add(nn.ReLU())
  g:add(nn.Dropout(params.dr))
  g:add(nn.Linear(params.hiddens,Dy))
  -- g:add(nn.BatchNormalization(Dy))
  g.p, g.dp = g:getParameters()

  -- discriminator(x,y)
  local f = nn.Sequential()
  f:add(nn.Linear(Dx+Dy,params.hiddens))
  f:add(nn.ReLU())
  f:add(nn.Dropout(params.dr))
  f:add(nn.Linear(params.hiddens,1))
  f:add(nn.Sigmoid())
  f.p, f.dp = f:getParameters()

  -- discriminator((x,generator(x,z)))
  local id_x = nn.Sequential():add(nn.Select(2,1)):add(nn.View(-1,1))
  local fg_concat = nn.ConcatTable():add(id_x):add(g)
  local fg = nn.Sequential():add(fg_concat):add(nn.JoinTable(2)):add(f)
  fg.p, fg.dp = g.p, g.dp

  -- labels and loss
  local y_real = torch.ones (params.bs,1):fill(0.8)
  local y_fake = torch.zeros(params.bs,1):fill(0.2)
  local ell    = nn.BCECriterion()

  local pf = {
    batchSize = params.bs,
    optimizer = optim.adam,
    optimState = {
      weightDecay = params.wd,
      learningRate = params.lr,
      beta1 = params.beta1,
      beta2 = params.beta2
    }
  }

  local pg = {
    batchSize = params.bs,
    optimizer = optim.adam,
    optimState = {
      weightDecay = params.wd,
      learningRate = params.lr,
      beta1 = params.beta1,
      beta2 = params.beta2
    }
  }

  local n_updates = 0

  for i=1,params.epochs do
    for b_x, b_y in metal.random_batches(x,y,params.bs) do
      local b_z  = torch.randn(b_x:size(1),params.dz)
      local b_xz = torch.cat(b_x, b_z, 2)
      local b_fake = torch.cat(b_x, metal.predict(g,b_xz), 2)
      local b_real = torch.cat(b_x, b_y, 2)
      if ((n_updates % 2) == 0) then
        -- train discriminator
        metal.train(f,ell,torch.cat(b_fake,b_real,1),torch.cat(y_fake,y_real,1),pf)
      else
        -- train generator
        metal.train(fg,ell,b_xz,y_real,pg)
      end
      n_updates = n_updates + 1
    end
  end

  local z = torch.randn(x:size(1), params.dz)
  return metal.predict(g,torch.cat(x,z,2))
end

local function plot(x,y,px,py,c_xy,c_yx)
  gnuplot.plot({'real', x:view(-1), y:view(-1), '+'},
               {'fake', x:view(-1), py:view(-1),'+'})
  gnuplot.title('Forward: ' .. c_xy)
  gnuplot.figure()
  gnuplot.plot({'real', x:view(-1), y:view(-1), '+'},
               {'fake', px:view(-1), y:view(-1),'+'})
  gnuplot.title('Backward: ' .. c_yx)
  io.read()
  gnuplot.closeall()
end

local function one_cause_effect(x,y,params)
  local py   = train_conditional_gan(x,y,params)
  local px   = train_conditional_gan(y,x,params)
  local c_xy, _ = test(torch.cat(x,y,2), torch.cat(x,py,2), params)
  local c_yx, _ = test(torch.cat(x,y,2), torch.cat(px,y,2), params)

  if ((params.my_test == 0) or (params.my_test == 1)) then
    c_xy = c_xy - 0.5
    c_yx = c_yx - 0.5
  end

  c_xy = math.abs(c_xy)
  c_yx = math.abs(c_yx)

  if (params.plot == 1) then plot(x,y,px,py,c_xy,c_yx) end

  return c_xy, c_yx
end

local function cause_effect(x,y,params)
  local c_xy = 0
  local c_yx = 0

  if (params.min == 1) then
    c_xy = 1e6
    c_yx = 1e6
  end

  for i=1,params.reps do
    local c_xy_i, c_yx_i = one_cause_effect(x,y,params)
    if (params.min == 1) then
      c_xy = math.min(c_xy, c_xy_i)
      c_yx = math.min(c_yx, c_yx_i)
    else
      c_xy = c_xy + c_xy_i/params.reps
      c_yx = c_yx + c_yx_i/params.reps
    end
  end

  return c_xy, c_yx
end

local meta = io.open(params.pairs_dir .. 'pairmeta.txt', 'r')
local total = 0
local correct = 0
local line = meta:read('*line')

while(line) do
  line = string.split(line, ' ')
  local fname  = params.pairs_dir .. 'pair' .. line[1] .. '.txt'
  local weight = tonumber(line[6])
  local parse  = false
  local label  = 0

  if ((line[2]=='1') and (line[3]=='1') and
      (line[4]=='2') and (line[5]=='2')) then
    parse = true
    label = 0
  end

  if ((line[2]=='2') and (line[3]=='2') and
      (line[4]=='1') and (line[5]=='1')) then
    parse = true
    label = 1
  end

  if (parse == true) then
    local x, y = read_pair(fname, params)
    local c_xy, c_yx = cause_effect(x, y, params)
    local result = ((((c_xy > c_yx) and 1 or 0) == label) and 1 or 0)*weight
    total = total + weight
    correct = correct + result
    print(string.format('%s %.5f %.5f %.5f %.5f %.5f %.5f', line[1], weight, label, result, c_xy, c_yx, correct/total))
  end

  line = meta:read('*line')
end
