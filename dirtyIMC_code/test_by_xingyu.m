function [errs] = test(k, d1, d2, n1, n2, sparsity, nModel)
% Tests IMC with a random synthetic dataset.
% Usage:
%	>> do_test_imc(k, d1, d2, n1, n2, m);
% Parameters:
%	k:	rank (default 5)
%	d1: number of row features (default 50)
%	d2: number of col features (default 50)
%	n1: number of rows (default 1000)
%	n2: number of cols (default 1000)
%	m:	nnz (default 1000)
%
% Author: Donghyuk Shin (dshin@cs.utexas.edu)

addpath(genpath('AFA_ALDM_publish'));

if nargin < 1, k  = 2;		end
if nargin < 2, d1 = 15;		end
if nargin < 3, d2 = 20;		end
if nargin < 4, n1 = 50;	end
if nargin < 5, n2 = 140;	end
if nargin < 6, sparsity  = 1;	end
if nargin < 7, nModel = 5;	end

trainFrac = 0.2;
randstate = 1;
lambda = 1e-1;
maxiter = 2000;
tol = 1e-5;

% generate random data
fprintf('Generating random data...');
[X,Y,A,Z] = Rank1_dep_setup(k,d1,d2,n1,n2,sparsity,randstate);
idx = find(A);
idx_test = datasample(idx, round(length(idx)*(1-trainFrac)), 'Replace', false);
trainMatrix = A; 
trainMatrix(idx_test) = 0;
fprintf('DONE!\n');

% run dirty imc
fprintf('L1\tL2\tErr\tRank(M)\tRank(N)\n')
loss = inf;
lambda = [10^-3 10^-2 10^-1 1 10 100];
lambda1 = [10^-3 10^-2 10^-1 1 10 100];
for a = 1:length(lambda)	% parameter selection
	for b = 1:length(lambda1)
		[UU SS VV U S V] = dirtyIMC(trainMatrix, X, Y, lambda(a), lambda1(b), 1);
		Completed = U*S*V'+X*UU*SS*VV'*Y';
		l = norm(Completed(idx_test)-A(idx_test), 'fro')/norm(A(idx_test), 'fro');
    tmp = Completed; tmp(idx_test) = 0;
    train_err = norm((tmp - trainMatrix), 'fro') / norm((trainMatrix), 'fro');
		fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', lambda(a), lambda1(b), l, rank(SS), rank(S));
    disp(train_err);
		if(l < loss)
			loss = l;
		end
	end
end
fprintf('dirtyIMC: \tloss = %f\n\n', loss)

end

% This is a function for generating synthetic IMC data.
function [X,Y,sparseMatrix,core] = Rank1_dep_setup(k,d1,d2,n1,n2,sparsity,randstate)
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

%rng(randstate);
core = sprand(d1,d2,sparsity);
X = sprand(n1,d1,sparsity);
Y = sprand(n2,d2,sparsity);

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

