% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% Written (W) 2012 Heiko Strathmann

% computes the diagonal of a Gaussian kernel matrix for given data (each datum
% one row) and a kernel size deg
function [H]=rbf_dot_diag(X,Y,deg);
n=size(X,1);

% for now assert that samples have same size
assert(n==size(Y,1));
% 
% H=zeros(n,1);
% 
% 
% for i=1:n
%     dist=X(i,:)-Y(i,:);
%     H(i)=exp(-(dist*dist')/2/deg^2);
% end
dists=X-Y;
dists=dists.*dists;
dists=dists';

% use sum over all columns, if there is only one row, prevent matlab from
% producing a single scalar
if size(dists,1)>1
    dists=sum(dists);
end

% precompute denominator
temp=2*deg^2;
    
    
H=exp(-dists/temp)';
