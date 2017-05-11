function [] = do_test_imc(k, d1, d2, n1, n2, m)
%
% Tests IMC with a random synthetic dataset.
%
% Usage:
%
%	>> do_test_imc(k, d1, d2, n1, n2, m);
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

if nargin < 1, k  = 5;		end
if nargin < 2, d1 = 50;		end
if nargin < 3, d2 = 50;		end
if nargin < 4, n1 = 1000;	end
if nargin < 5, n2 = 1000;	end
if nargin < 6, m  = 1000;	end

randstate = 1;

lambda = 1e-6;
maxiter = 10;

% generate random data
fprintf('Generating random data...');
[X,Y,A,Z] = Rank1_dep_setup(k,d1,d2,n1,n2,m,randstate);
[I,J,K] = find(A);
M = sparse(I,J,true,n1,n2);
W0 = randn(d1, k); H0 = randn(d2, k);
fprintf('DONE!\n');

% run IMC
[W, H] = IMC(A, X, Y, k, lambda, maxiter, W0, H0);

relerr = norm(W'*H-Z,'fro')^2 / norm(Z,'fro')^2;
fprintf('RelErr = %e\n',relerr);

end




% This is a function for generating synthetic IMC data.
function [X,Y,G,W] = Rank1_dep_setup(k,d1,d2,n1,n2,m,randstate)
% Input:
% k--rank
% d1,d2--dimension of latent space
% n1,n2--user,movie sizes
% m--number of samples
% rand seed

% Output:
% X, Y are features.
% W is the ground truth low rank matrix.
% G is a sparse n1*n2 matrix, where each nonzero entry is an observed sample.
% G = (X*W*Y')_\Omega, where \Omega is the sample set with size m.

rng(randstate);
W = randn(d1,k)*randn(k,d2);

X = randn(n1,d1);
Y = randn(n2,d2);

% Orthogonalize X and Y
[X,R_X] = qr(X,0);
[Y,R_Y] = qr(Y,0);

Omega = randsample(n1*n2,m);
ii = zeros(m,1);
jj = zeros(m,1);
b = zeros(m,1);
for ij=1:m
    i = floor((Omega(ij)-1)/n2)+1;
    ii(ij) = i;
    j = Omega(ij) - (i-1)*n2;
    jj(ij) = j;
    b(ij) = X(i,:)*W*Y(j,:)';
end

G = sparse(ii,jj,b,n1,n2);

end


