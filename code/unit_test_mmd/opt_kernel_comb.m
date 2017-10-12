function [bg] = opt_kernel_comb(X,Y)

sigmas=2.^[-15:1:10];
lss=length(sigmas);

% some local variables to make code look nicer
m=size(X,1);
m2=ceil(m/2);

% preallocate arrays for mmds, ratios, etc
mmds=zeros(lss,1);
vars=zeros(lss,1);
ratios=zeros(lss,1);
hh=zeros(lss,m2);

% single kernel selection methods are evaluated for all kernel sizes
for i=1:lss
    % compute kernel diagonals
    K_XX = rbf_dot_diag(X(1:m2,:),X(m2+1:m,:),sigmas(i));
    K_YY = rbf_dot_diag(Y(1:m2,:),Y(m2+1:m,:),sigmas(i));
    K_XY = rbf_dot_diag(X(1:m2,:),Y(m2+1:m,:),sigmas(i));
    K_YX = rbf_dot_diag(X(m2+1:m,:),Y(1:m2,:),sigmas(i));
    
    % this corresponds to the h-statistic that the linear time MMD is the
    % average of
    hh(i,:)=K_XX+K_YY-K_XY-K_YX;
    mmds(i)=mean(hh(i,:));
    
    %variance computed using h-entries from linear time statistic
    vars(i)=var(hh(i,:));
    
    % add lambda to ensure numerical stability
    ratios(i)=mmds(i)/(sqrt(vars(i))+10E-4);
    
    % always avoid NaN as the screw up comparisons later. The appear due to
    % divisions by zero. This effectively makes the test fail for the
    % kernel that produced the NaN
    ratios(isnan(ratios))=0;
    ratios(isinf(ratios))=0;
end

% kernel selection method: maxrat
% selects a single kernel that maxismised ratio of the MMD by its standard
% deviation. This leads to optimal kernel selection
w_maxrat=zeros(lss,1);
[~,idx_maxrat]=max(ratios);
w_maxrat(idx_maxrat)=1;
bg = sigmas(idx_maxrat);
