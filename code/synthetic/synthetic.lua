local randomkit = require 'randomkit'

local cmd = torch.CmdLine()
cmd:option('--alpha',0.05,'significance level')
cmd:option('--my_test',0,'[knn|neural]')
cmd:option('--seed',0,'random seed')
local params = cmd:parse(arg)

local test
if (params.my_test == 0) then
  test = require '../common/knn.lua'
elseif (params.my_test == 1) then
  test = require '../common/neural.lua'
end

local function normalize(x)
  local y = x:clone()
  local m = x:mean(1)
  local s = x:std(1)
  y:add(m:mul(-1):expandAs(y)):cdiv(s:expandAs(y))
  return y
end

local function experiment_student(ename, n, d, df)
  local p = normalize(torch.randn(n,d))
  local q = normalize(randomkit.standard_t(torch.randn(n,d),df))
  local test_m, test_t = test(p,q)
  if (test_m ~= nil) then
    print(params.my_test, ename,n,d,df,test_m,test_t,(test_t <= params.alpha) and 1 or 0)
  end
end

local function experiment_sinusoid(ename, n, freq, noise)
  local x  = normalize(torch.randn(n,1))
  local yp = normalize(torch.add(torch.cos(torch.mul(x,freq)), torch.randn(n,1):mul(noise)))
  local yq = yp:index(1,torch.randperm(yp:size(1)):long())
  local p = torch.cat(x,yp,2)
  local q = torch.cat(x,yq,2)

  local test_m, test_t = test(p,q)
  if (test_m ~= nil) then
    print(params.my_test,ename,n,freq,noise,test_m,test_t,(test_t <= params.alpha) and 1 or 0)
  end
end

local function experiment_type1(ename, n, d)
  local p = torch.randn(n,d)
  local q = torch.randn(n,d)
  local test_m, test_t = test(p,q)
  if (test_m ~= nil) then
    print(params.my_test, ename, n, d, -1, test_m, test_t, (test_t > params.alpha) and 1 or 0)
  end
end

for rep=1,10 do
  experiment_student('student_df', 2000, 1, 10)
end

local grid_n = { 50, 100, 500, 1000, 2000 }
for ni=1,#grid_n do
  experiment_student('student_n', grid_n[ni], 1, 3)
end

local grid_df = { 1, 2, 5, 10, 15, 20 }
for dfi=1,#grid_df do
  experiment_student('student_df', 2000, 1, grid_df[dfi])
end

local grid_n = { 50, 100, 500, 1000, 2000 }
for ni=1,#grid_n do
  experiment_sinusoid('sinusoid_n', grid_n[ni], 1, 0.25)
end

local grid_f = { 2, 4, 6, 8, 10, 20 }
for fi=1,#grid_f do
  experiment_sinusoid('sinusoid_f', 2000, grid_f[fi], 0.25)
end

local grid_v = { 0.1, 0.25, 0.5, 1, 2, 3 }
for vi=1,#grid_v do
  experiment_sinusoid('sinusoid_v', 2000, 1, grid_v[vi])
end

local grid_n = { 50, 100, 500, 1000, 2000 }
for ni=1,#grid_n do
  experiment_type1('type1_n', grid_n[ni],1)
end
