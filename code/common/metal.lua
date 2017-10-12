-------------------------------------------------------------------------------
-- metal, easily train and evaluate neural networks in torch
--
-- Copyright (c) 2016 Facebook (David Lopez-Paz)
-- 
-- All rights reserved.
-- 
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
-- 
-- 1. Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
-- 
-- 2. Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
-- 
-- 3. Neither the names of NEC Laboratories American and IDIAP Research
--    Institute nor the names of its contributors may be used to endorse or
--    promote products derived from this software without specific prior
--    written permission.
-- 
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
-- AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
-- ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
-- LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
-- CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
-- SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
-- INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
-- CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
-- ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
-- POSSIBILITY OF SUCH DAMAGE.
-- 
-------------------------------------------------------------------------------

local metal = {}

local nn = require 'nn'
local xlua = require 'xlua'
local optim = require 'optim'

function metal.get_rows(x,idx)
  if (torch.type(x) == 'table') then
    local res = {}
    for i = 1,#x do
      res[#res+1] = metal.get_rows(x[i],idx)
    end
    return res
  end
  return x:index(1,idx)
end

function metal.random_batches(x,y,bs)
  local bs = bs or 1
  local n

  if (torch.type(x) == 'table') then
    n = x[1]:size(1)
  else
    n = x:size(1)
  end

  local p  = torch.randperm(n):long()
  local index = 1
  local finish = false

  return function ()
    if (finish == true) then
      return nil
    end

    local from  = index
    local to    = math.min(index + bs - 1, n)
    index = index + bs

    if (to == n) then
      finish = true
    end

    if to <= n then
      return metal.get_rows(x, p[{{from,to}}]),
             metal.get_rows(y, p[{{from,to}}])
    end
  end
end

function metal.normalize(x, eps)
  local eps = eps or 0
  local y = x:clone()
  local m = x:mean(1):mul(-1)
  local s = x:std(1)+eps
  y:add(m:expandAs(y)):cdiv(s:expandAs(y))
  return y
end

function metal.train_test_split(x,y,p_tr)
  local p_tr = p_tr or 0.5
  local n_tr = math.floor(x:size(1)*p_tr)
  local i = torch.randperm(x:size(1)):long()
  local i_tr = i[{{1,n_tr}}]
  local i_te = i[{{n_tr+1,x:size(1)}}]
  local x_tr = metal.get_rows(x,i_tr)
  local y_tr = metal.get_rows(y,i_tr)
  local x_te = metal.get_rows(x,i_te)
  local y_te = metal.get_rows(y,i_te)
  return x_tr, y_tr, x_te, y_te
end

function metal.train(net, ell, x, y, parameters)
  local parameters = parameters or {}                 -- the parameters are:
  local gpu = parameters.gpu or false                 -- use GPU? 
  local verbose = parameters.verbose or false         -- display progress?
  local batchSize = parameters.batchSize or 64        -- batch size
  local optimizer = parameters.optimizer or optim.sgd -- optimizer 
  local optimState = parameters.optimState or {}      -- optimizer state

  -- first call... 
  if (net.p == nil) then
    if gpu then net:cuda() end
    if gpu then ell:cuda() end
    net.p, net.dp = net:getParameters()
  end
  
  -- set net in training mode
  net:training()
  net:zeroGradParameters()
 
  -- nn.ClassNLLCriterion does not work with 2D targets
  if (ell.__typename == 'nn.ClassNLLCriterion') then
    y = y:view(-1)
  end

  -- these are the "global" input/target variables
  local input, target

  -- optimize function handle
  local function handle(x)
     net.dp:zero()
     local prediction = net:forward(input)
     local loss = ell:forward(prediction, target)
     local gradient = ell:backward(prediction, target)
     net:backward(input, gradient)
     return loss, net.dp 
  end

  -- get number of rows in tensor/table
  local n
  if (torch.type(x) == 'table') then
    n = x[1]:size(1)
  else
    n = x:size(1)
  end

  -- proceed in minibatches
  for bx, by in metal.random_batches(x,y,batchSize) do
    input = bx
    target = by
    if gpu then input = input:cuda() end
    if gpu then target = target:cuda() end 
    -- train
    optimizer(handle, net.p, optimState)
    -- report progress if verbose 
    if (verbose == true) then
      xlua.progress(i,n)
    end
  end
end

function metal.predict(net, x, parameters)
  local parameters = parameters or {}
  local batchSize = parameters.batchSize or 16
  local gpu = parameters.gpu or false
  local verbose = parameters.verbose or false
  local predictions

  net:evaluate()
  
  -- get number of rows in tensor/table
  local n
  if (torch.type(x) == 'table') then
    n = x[1]:size(1)
  else
    n = x:size(1)
  end

  for i=1,n,batchSize do
    local to = math.min(i+batchSize-1,n)
    local idx = torch.range(i,to):long()
    local input = metal.get_rows(x,idx)
    if gpu then input = input:cuda() end
    if gpu then target = target:cuda() end 
    local prediction = net:forward(input)

    if(torch.type(prediction) == 'table') then
      if (predictions == nil) then
        predictions = {}
        for j=1,#prediction do
          if (prediction[j]:dim() == 1) then
            predictions[j] = torch.zeros(n)
          else
            predictions[j] = torch.zeros(n, prediction[j]:size(2))
          end
        end
      end

      for j=1,#prediction do
        predictions[j][{{i,to}}] = prediction[j]
      end
    else
      if (predictions == nil) then
        if (prediction:dim() == 1) then
          predictions = torch.zeros(n)
        else
          predictions = torch.zeros(n, prediction:size(2))
        end
      end
      predictions[{{i,to}}] = prediction
    end
  end
 
  return predictions
end

function metal.eval(net, ell, x, y, parameters)
  local parameters = parameters or {}
  local batchSize = parameters.batchSize or 16
  local gpu = parameters.gpu or false
  local verbose = parameters.verbose or false

  local predictions = metal.predict(net, x, parameters)
  local accuracy

  if (ell.__typename == 'nn.BCECriterion') then
    local plabels = torch.ge(predictions,0.5):long()
    accuracy = torch.eq(plabels,y:long()):double():mean()
  end

  if ((ell.__typename == 'nn.ClassNLLCriterion') or
      (ell.__typename == 'nn.CrossEntropyCriterion')) then
    local _, plabels = torch.max(predictions,2)
    accuracy = torch.eq(plabels,y:view(-1):long()):double():mean()
  end
  
  if (accuracy) then
    return ell:forward(predictions, y), accuracy
  else
    return ell:forward(predictions, y)
  end
end

function metal.save(net, fname)
  net:evaluate()
  net:clearState()
  net.p = nil
  net.dp = nil
  torch.save(fname, net)
end

function metal.load(fname)
  return torch.load(fname)
end

return metal
