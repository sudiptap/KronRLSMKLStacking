function [ A ] = kronrls_mkboost( K1, K2, y, lambda, regcoef, maxiter, isinner, num_ones)
%KRONRLS_MKL Summary of this function goes here
%   INPUTS
%    'K1' : 3 dimensional array of kernels
%    'K2' : 3 dimensional array of kernels
%    'y'  : adjacency matrix
%    'lambda' : lambda regulaziation coefficient of KronRLS algorithm
%    'regcoef': regularization parameter of kernel weights
%    'maxiter': maximum number of iterations (default=20)
%
%   OUTPUTS
%    'y2': predictions
%    'alpha': weights of kernels K1
%    'beta':  weights of kernels K2

    %restrict to MKL on both K1 and K2
    assert(size(K1,3)>1 && size(K2,3)>1,'K1 and K2 must order 3 tensors');
    
    if ~exist('maxiter','var') || isempty(maxiter)
        maxiter = 20;
    end
    if ~exist('regcoef','var') || isempty(regcoef)
        regcoef = 0.2;
    end
    if ~exist('isinner','var') || isempty(isinner)
        isinner = 0;
    end
    
    nA = size(K1,3); 
    nB = size(K2,3); 
    
    %initialization of kernel weights (uniform)
    alpha(1:nA) = 1/nA;
    beta(1:nB)  = 1/nB;
  
    iter = 0;
    incr = 1000;
    limit_incr = 0.0100;
    combFval = zeros(1,maxiter);
    
    if ~isinner
        regcoef = optimize_regcoef(K1, K2, y);
    end

    
    
    K2_comb = combine_kernels(beta, K2);
    %% iterative steps: optimize weights
    %while(iter < maxiter && abs(incr) > limit_incr)      
    while(iter < maxiter)
	[dist] = init_distribution(y, num_ones);      
        iter = iter+1;
        
	%% Step 1
	K1_comb = boost_kernels(y, K1, K2_comb, dist, true, lambda);
	K2_comb = boost_kernels(y, K2, K1_comb, false, lambda);
	
	%if ~isinner
	%    fprintf('ITER=%d \tlambda=%.2f \tregcoef=%.1f - (fval_alpha=%.4f, fval_beta=%.4f) \n',iter, lambda, regcoef, fval_alpha, fval_beta)
	%end	
	
    end
    
    %% Perform predictions
    %K1_comb = combine_kernels(alpha, K1);
    %K2_comb = combine_kernels(beta, K2);
    
    %build final model with best weights/lambda
    %if ~isinner
    %    lambda  = optimize_lambda(K1_comb, K2_comb, y);
    %end
    [A] = kronrls(K1_comb, K2_comb, y, lambda, num_ones);

end

function [dist] = init_distribution(y, num_ones)
    %initialize the distributions
    %y is the training set. we have one weight associated with each record in training set    
    dist = zeros(size(y));
    %for kc_i = 1:kernelCombsLen
    for dist_i = 1:size(dist,1)
	for dist_j = 1:size(dist,2)
	    dist(dist_i, dist_j) = 1/num_ones;
	end
    end
end 

function [] = manage_kernels(iter)
    if iter==1
        K1_comb = combine_kernels(alpha, K1);
	K2_comb = combine_kernels(beta, K2);
    else
        K1_comb = boost_kernels(alpha, K1);
	K2_comb = boost_kernels(beta, K2);
    end    
end

function [kernel_comb] = boost_kernels(y_train, K, K_comb, dist, k1_k2, lambda)
    [alpha, best_kernel] = mkboost_d1_mplusn(y_train, K, K_comb, dist, k1_k2, lambda);
    kernel_comb = zeros(size(K(:,:,1)));
    %boost the kernels using mkboost_d1 output  
    for alpha_i=1:size(alpha)  
	disp(best_kernel(alpha_i));
        kernel_comb = kernel_comb + alpha(alpha_i) .* K(:,:,best_kernel(alpha_i));
    end
end



