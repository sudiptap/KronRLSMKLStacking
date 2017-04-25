function [] = xingyu_test(k, d1, d2, n1, n2, m)
%
% Tests IMC (Matlab wrapper) with a randomly generated synthetic dataset.
%
% Usage:
%	>> do_test_imf(k, d1, d2, n1, n2, m);
%
% Parameters:
%	k:	rank (default 5)
%	d1: number of row features (default 50)
%	d2: number of col features (default 50)
%	n1: number of rows (default 1000)
%	n2: number of cols (default 1000)
%	m:	nnz (default 1000)
%
%
% Author: Donghyuk Shin (dshin@cs.utexas.edu)
%

addpath('../matlab');

% if nargin < 1, k = 5; end
% if nargin < 2, d1 = 50; end
% if nargin < 3, d2 = 50; end
% if nargin < 4, n1 = 1000; end
% if nargin < 5, n2 = 1000; end
% if nargin < 6, m = 1000; end
% 
% rng(0);
% 
% threads = 1;
% lambda = 1e-6;
% maxiter = 10;
% imcOpt = sprintf('-n %d -k %d -l %s -t %d',threads,k,num2str(lambda),maxiter);
% 
% 
% % generate random data
% fprintf('Generating synthetic data...');
% [X,Y,A,Z] = Rank1_dep_setup(k,d1,d2,n1,n2,m);
% W0 = randn(k, d1); H0 = randn(k, d2);
% fprintf('DONE!\n');
% 
% % run IMC
% [W, H, wtime] = train_mf(A,X,Y,W0,H0,imcOpt);
% 
% relerr = norm(W'*H-Z,'fro')^2 / norm(Z,'fro')^2;
% fprintf('RelErr = %e  Time = %.4f sec\n',relerr,wtime);


if nargin < 1, k  = 3;    end             
if nargin < 2, d1 = 100;   end             
if nargin < 3, d2 = 100;   end             
if nargin < 4, n1 = 100;  end             
if nargin < 5, n2 = 100;  end             
if nargin < 6, sparsity  = 0.2; end       

threads = 1;                                          
trainFrac = 0.9;                          
randstate = 1;                            
lambda = 1e-1;                            
maxiter = 20;                           
tol = 1e-5;                              
nModel = 5;
imcOpt = sprintf('-n %d -k %d -l %s -t %d',threads,k,num2str(lambda),maxiter);
                                          
% generate random data                    
fprintf('Generating random data...');     
[X,Y,A,Z] = Rank1_dep_setup(k,d1,d2,n1,n2,sparsity,randstate);  
idx = find(A);                                                  
idx_test = datasample(idx, round(length(idx)*(1-trainFrac)), 'Replace', false); 
trainMatrix = sparse(A);                            
trainMatrix(idx_test) = 0;                         
fprintf('DONE!\n');                   
                                                   
% run IMC                                          
tic;                                               
k_imc = nModel*k;                                  
W0 = randn(k_imc, d1); H0 = randn( k_imc, d2);      
[W, H] = train_mf(trainMatrix, X, Y,  W0, H0, imcOpt); 
toc;                                              
estimate = X*(W')*H*Y';                           
relerr = norm(estimate(idx_test) - A(idx_test),'fro') / norm(A(idx_test),'fro'); 
fprintf('RelErr = %e\n', relerr);

end



% This is a function for generating synthetic IMC data.
function [X,Y,sparseMatrix, core] = Rank1_dep_setup(k,d1,d2,n1,n2,sparsity,randstate)
% Input:
% k--rank
% d1,d2--dimension of latent space
% n1,n2--user,movie sizes
% m--number of samples

% Output:
% X, Y are features.
% W is the ground truth low rank matrix.
% G is a sparse n1*n2 matrix, where each nonzero entry is an observed sample.
% G = (X*W*Y')_\Omega, where \Omega is the sample set with size m.


% W = randn(d1,k)*randn(k,d2);
% 
% X = randn(n1,d1);
% Y = randn(n2,d2);
% 
% %Orthogonalize X and Y
% [X,R_X] = qr(X,0);
% [Y,R_Y] = qr(Y,0);
% 
% Omega = randsample(n1*n2,m);
% ii = zeros(m,1);
% jj = zeros(m,1);
% b = zeros(m,1);
% for ij=1:m
%     i = floor((Omega(ij)-1)/n2)+1;
%     ii(ij) = i;
%     j = Omega(ij) - (i-1)*n2;
%     jj(ij) = j;
%     b(ij) = X(i,:)*W*Y(j,:)';
% end
% 
% G = sparse(ii,jj,b,n1,n2);
% end

core = round(sprand(d1,d2,sparsity).*5); 
X = round(sprand(n1,d1,sparsity).*5);    
Y = round(sprand(n2,d2,sparsity).*5);    
                               
% Orthogonalize X and Y        
%[X,R_X] = qr(X,0);            
%[Y,R_Y] = qr(Y,0);            
                               
[y, idx] = datasample([1:n1*n2], round(n1*n2*sparsity), 'Replace', false);  
M = X*core*Y';   
M = (M - min(min(M)))/(max(max(M)) - min(min(M)));   
sparseMatrix = zeros(n1,n2);                         
sparseMatrix(idx) = M(idx);                          
return;                                              
end

