require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

local cmd = torch.CmdLine()
cmd:option('--my_test','mmd')
cmd:option('--seed',0)
cmd:option('--base_dir', '.')
cmd:option('--dataset', 'faces')
cmd:option('--features', 'pixels')
cmd:option('--collage', 50)
cmd:option('--gf', 32)
cmd:option('--df', 32)
cmd:option('--ep', 200)
cmd:option('--max_n', 100000)
cmd:option('--bs', 128)
local params = cmd:parse(arg)

torch.manualSeed(params.seed)

local function featurize(net, imgs)
  local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std  = { 0.229, 0.224, 0.225 },
  }

  local predictions = torch.zeros(imgs:size(1), 2048)

  for n=1,imgs:size(1) do
    img = imgs[n]
    for i=1,3 do
      img[i]:add(-meanstd.mean[i])
      img[i]:div(meanstd.std[i])
    end
  end

  for n=1,imgs:size(1),params.bs do
    to = math.min(imgs:size(1), n+params.bs-1)
    predictions[{{n,to}}] = net:forward(imgs[{{n,to}}]:cuda()):float():clone()
  end

  return predictions
end

local test
if (params.my_test == 'mmd') then
  test = dofile '../common/mmd.lua'
elseif (params.my_test == 'knn') then
  test = dofile '../common/knn.lua'
elseif (params.my_test == 'neural') then
  test = dofile '../common/neural.lua'
end

local real_dir = params.base_dir .. 'fixedDatasets/' .. params.dataset .. '.t7' -- https://github.com/soumith/dcgan.torch
local real_samples = torch.load(real_dir)
print('Read ' ..  real_samples:size(1) .. ' real ' .. params.dataset)
local p = torch.randperm(real_samples:size(1))[{{1,params.collage}}]:long()
-- image.save(real_dir .. '.jpg', image.toDisplayTensor{input=real_samples:index(1,p), nrow=params.collage})
print (real_dir .. '.jpg')

local fake_file = params.dataset .. '_g' .. params.gf .. '_d' .. params.df .. '_ep' .. params.ep .. '_generator.t7'
local fake_dir = params.base_dir .. params.dataset .. '/savedGenImages/' .. fake_file
local fake_samples = torch.load(fake_dir)
fake_samples = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(-1,real_samples:size(2),real_samples:size(3),real_samples:size(4))):forward(fake_samples)
print('Read ' ..  fake_samples:size(1) .. ' fake ' .. params.dataset)
local p = torch.randperm(fake_samples:size(1))[{{1,params.collage}}]:long()
-- image.save(fake_dir .. '.jpg', image.toDisplayTensor{input=fake_samples:index(1,p), nrow=params.collage})
print (fake_dir .. '.jpg')

if (real_samples:size(1) > fake_samples:size(1)) then
  local n = math.min(fake_samples:size(1), params.max_n)
  real_samples = real_samples:index(1,torch.randperm(real_samples:size(1))[{{1,n}}]:long()):float()
  fake_samples = fake_samples:index(1,torch.randperm(fake_samples:size(1))[{{1,n}}]:long()):float()
else
  local n = math.min(real_samples:size(1), params.max_n)
  real_samples = real_samples:index(1,torch.randperm(real_samples:size(1))[{{1,n}}]:long()):float()
  fake_samples = fake_samples:index(1,torch.randperm(fake_samples:size(1))[{{1,n}}]:long()):float()
end

if (params.features == 'resnet') then
  local net = torch.load(params.base_dir .. 'resNet/resnet.t7') -- https://github.com/facebook/fb.resnet.torch
  net:remove()
  net:remove()
  net:remove()
  if false then
    net:add(cudnn.SpatialAveragePooling(2,2))
    net:add(nn.View(512))
  else
    net:add(nn.View(2048))
  end
  net:evaluate()
  net:cuda()
  torch.save('little_' .. params.dataset .. '_pixels.t7', { real_samples, fake_samples })
  real_samples = featurize(net, real_samples)
  fake_samples = featurize(net, fake_samples)
  torch.save('little_' .. params.dataset .. '_resnet.t7', { real_samples, fake_samples })
else
  real_samples = real_samples:view(real_samples:size(1),-1):clone()
  fake_samples = fake_samples:view(fake_samples:size(1),-1):clone()
end

print(params.my_test, params.features, params.dataset, params.gf, params.df, params.ep, test_m, test_t)
