function [ A ] = kronrls_pairwise_mkboost( K1, K2, y, lambda, regcoef, maxiter, isinner, num_ones)
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
    %alpha(1:nA) = 1/nA;
    %beta(1:nB)  = 1/nB;
  
    iter = 0;
    incr = 1000;
    limit_incr = 0.0100;
    combFval = zeros(1,maxiter);
    
    %if ~isinner
    %    regcoef = optimize_regcoef(K1, K2, y);
    %end

    kernelCombsLen = nA * nB;
    kernelCombs = cell(1, kernelCombsLen);
    index=1;
    for nAi = 1:nA
	for nBi=1:nB	
	   temp = cell(1,2);
           temp{1} = K1(:,:,nAi);   
           temp{2} = K2(:,:,nBi);
           kernelCombs{index} = temp; index = index + 1;
        end
    end

    %initialize the distributions
    %y is the training set. we have one weight associated with each record in training set    
    dist = zeros(size(y)); 
     
    for dist_i = 1:size(dist,1)
		for dist_j = 1:size(dist,2)
			%disp(num_ones);
			if(y(dist_i, dist_j)==1)
				dist(dist_i, dist_j) = 1/num_ones;	
			end
			%disp(num_ones);
			%dist(dist_i, dist_j) = 1/(size(dist,1)*size(dist,2));
		end
    end
    %disp('calling');
    %[alpha, best_kernel_index] = mkboost_d1(y, kernelCombs, dist, lambda, num_ones);
    [A] = mkboost_d2(y, kernelCombs, dist, lambda, num_ones);
    %disp('called');
    %A = zeros(size(y));
    
    %for alpha_i=1:size(alpha)
    %	temp = kernelCombs{alpha_i}; 
    %    [A] = [A] + alpha(alpha_i) .* kronrls(temp{1}, temp{2}, y, lambda);
	
    %end
    %disp(size(A));
	%return;
end

function [x, fval] = optimize_weights(x0, fun)
    n = length(x0);
    Aineq   = [];
    bineq   = [];
    Aeq     = ones(1,n);
    beq     = 1;
    LB      = zeros(1,n);
    UB      = ones(1,n);

    options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'notify');
    [x,fval] = fmincon(fun,x0,Aineq,bineq,Aeq,beq,LB,UB,[],options);
end

function [J] = obj_function(w,u,Ma,lambda,regcoef)
    V = combine_kernels(w, Ma);
    J = norm(u-V','fro')/(2*lambda*numel(V)) + regcoef*(norm(w,2))^2;
end
