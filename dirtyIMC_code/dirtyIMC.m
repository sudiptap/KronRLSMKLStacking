% M = UU*SS*VV', N = U*S*V', model = XMY^T + N

function [UU SS VV U S V] = dirtyIMC(A, X, Y, lambda, lambda1, maxit, showopt)
	if ~exist('maxit')
		maxit = 5;
	end
	if ~exist('showopt')
		showopt = 0;
	end
	
	[ii jj vv] = find(A);
	[n1, d1] = size(X);
	[n2, d2] = size(Y);
	if(size(A, 1) ~= n1 || size(A, 2) ~= n2)
		error('Dimensionality between observations and features does not match\n');
	end

	% M = UU*DD*UU'
	UU = zeros(d1, 1);
	SS = zeros(1, 1);
	VV = zeros(d2, 1);

	% N = U*S*V'
	U = zeros(n1, 1);
	S = zeros(1, 1);
	V = zeros(n2, 1);

	for i = 1:maxit

		% fix N, solve M with IMC
		if(showopt ~= 0)
			fprintf('Iteration %d:\n', i);
			fprintf('\tFix N, solve M with IMC...\t');
		end
		%entries = vv - dotp(U*S, V, ii, jj);
    tmp = U*S;
    entries = vv - dot(tmp(ii,:), V(jj,:), 2);
		A = sparse(ii, jj, entries, n1, n2);
		[UU SS VV] = mysolver_IMC(A, X, Y, lambda, 5, UU, SS, VV, 0);

		%obj = sum((vv-dotp(U*S, V, ii, jj)+dotp(X*UU*SS*VV', Y, ii, jj)).^2)/2 + lambda*sum(diag(SS))+lambda1*sum(diag(S));
    tmp = U*S; tmp1 = X*UU*SS*VV';
		obj = sum((vv-dot(tmp(ii,:), V(jj,:), 2)+dot(tmp1(ii,:), Y(jj,:), 2)).^2)/2 + lambda*sum(diag(SS))+lambda1*sum(diag(S));
		if(showopt ~= 0)
			fprintf('Objective = %e\n', obj);
		end


%    %% call AFA_ALDM
%    maxIter = 3000;
%    eps = 10e-6;
%    zeta = 10e6;
%    % ----- tune params
%    seq_lambda_E_aldm = [ 1:2:10];
%    seq_lambda_G_aldm = [0.0001: 0.0005:0.001];
%    training_error_matrix = zeros( length( seq_lambda_E_aldm), length( seq_lambda_G_aldm));
%    for i = 1 : length( seq_lambda_E_aldm )
%        for j = 1: length( seq_lambda_G_aldm)
%            [data_after_dump, dump_position] = selectdata( A, 0.1);
%            [G_new_dump, E_new_dump] =  AFA_ALDM(X', Y', A,  seq_lambda_E_aldm(i),  seq_lambda_G_aldm(j), maxIter, eps,zeta); 
%            training_error_matrix( i, j) =  norm( E_new_dump( dump_position) - A( dump_position))/norm(A( dump_position));
%        end
%    end
%    % ----- get relative error matrix 
%    index = find( training_error_matrix == min( min( training_error_matrix )));
%    [I_row, I_col] = ind2sub(size( training_error_matrix), index);
%    lambda_E = seq_lambda_E_aldm( I_row );
%    lambda_G = seq_lambda_G_aldm( I_col );
%    % ----- call AFA_ALDM
%    [G_new, E_new] = AFA_ALDM(X', Y', A,  lambda_E, lambda_G, maxIter, eps, zeta);


		% fix M, solve N with nuclear norm minimization
		if(showopt ~= 0)
			fprintf('\tFix M, solve N with MC...\t');
		end
		%entries = vv - dotp(X*UU*SS*VV', Y, ii, jj);
    tmp = X*UU*SS*VV'; 
    % add
    % tmp = X*G_new;
    entries = vv - dot(tmp(ii,:), Y(jj,:), 2);
		B = sparse(ii, jj, entries, n1, n2);
		[U S V] = mysolver_alt(B, lambda1, [], min(d1, d2), 5, U, S, V, 0);

		%obj = sum((vv-dotp(U*S, V, ii, jj)+dotp(X*UU*SS*VV', Y, ii, jj)).^2)/2 + lambda*sum(diag(SS))+lambda1*sum(diag(S));
    tmp = U*S; tmp1 = X*UU*SS*VV';
		obj = sum((vv-dot(tmp(ii,:), V(jj,:), 2)+dot(tmp1(ii,:), Y(jj,:), 2)).^2)/2 + lambda*sum(diag(SS))+lambda1*sum(diag(S));
		if(showopt ~= 0)
			fprintf('Objective = %e\n', obj);
		end

	end

end
