function kronrls_mkstack1( K1, K2, y, train_idx, lambda, num_ones, y_test, test_idx, y_orig)
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

  addpath 'dirtyIMC_code';
  addpath('imc-matlab');


	assert(size(K1,3)>1 && size(K2,3)>1,'K1 and K2 must order 3 tensors');
    %disp(nnz(y_test));
    if ~exist('maxiter','var') || isempty(maxiter)
        maxiter = 20;
    end
    if ~exist('regcoef','var') || isempty(regcoef)
        regcoef = 0.2;
    end
    if ~exist('isinner','var') || isempty(isinner)
        isinner = 0;
    end
    %disp(size(train_idx,1));
	%disp(size(test_idx,1));
	%get the number of kernels
    nA = size(K1,3); 
    nB = size(K2,3); 
	stacking_rows = size(y,1)*size(y,2);
	stacking_cols = nA*nB;
	out = zeros(size(train_idx,1), stacking_cols); disp(size(out));
	out_test = zeros(size(test_idx,1), stacking_cols); disp(size(out_test));
	%record all results for further analysis
	%prediction = cell(size(K1,1)*size(K1,2));
	%prediction = zeros(:,:);
	%tot_entires = size(K1,1)*size(K1,2);
	new_col_idx = 1; 
	orig = reshape(y_orig, [1, size(y_orig,1)*size(y_orig,2)]);
	%orig_test = reshape(y_test, [1, size(y_test,1)*size(y_test,2)]);
	label_train = zeros(1,size(train_idx,1));
	label_test = zeros(1,size(test_idx,1));
	for i=1:nA
		for j=1:nB
      d1 = size(y,1); d2 = size(y,2);
      k = min(d1,d2);
      [X,a1,v1] = svds(K1(:,:,i), d1);
      [Y,a2,v2] = svds(K2(:,:,j), d2);

      %% dirty imc
      %lambda = [10^-3 10^-2 10^-1 1 10 100];
      %lambda1 = [10^-3 10^-2 10^-1 1 10 100];
      %loss = 1000;
      %for a = 1:length(lambda)	% parameter selection
      %	for b = 1:length(lambda1)
      %		[UU SS VV U S V] = dirtyIMC(y, X, Y, lambda(a), lambda1(b));
      %		Completed = U*S*V'+X*UU*SS*VV'*Y';
      %		l = norm(Completed-y_orig, 'fro')/norm(y_orig, 'fro');
      %		fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', lambda(a), lambda1(b), l, rank(SS), rank(S));
      %		if(l < loss)
      %			loss = l; 
      %      pred = Completed;
      %		end
      %	end
      %end
      %norm(pred(test_idx)-y_orig(test_idx), 'fro')/norm(y_orig(test_idx), 'fro')
      %pred(test_idx)
      %y_orig(test_idx)
 
      %% imc
      threads = 1;
      lambda = 1e-6;
      maxiter = 100;
      d1 = size(y,1); d2 = size(y,2);
      k = min(d1,d2);
      imcOpt = sprintf('-n %d -k %d -l %s -t %d',threads,k,num2str(lambda),maxiter);
      % run IMC
      %[W, H, wtime] = train_mf(sparse(y),X,Y,W0,H0,imcOpt);
      [W, H] = IMC(y, X, Y, k, lambda, maxiter);
      relerr = norm(W'*H-y_orig,'fro')^2 / norm(y_orig,'fro')^2;
      fprintf('RelErr = %e \n',relerr);
      pred = X*W'*H*Y';
			
			pred1 = reshape(pred, [1, size(y,1)*size(y,2)]);
			%pred_test = pred_train(test_idx);			
			
			%disp(size(pred1));
			x_train_idx=1; x_test_idx=1;
			for x=1:stacking_rows
				if ismember(x,train_idx)%orig(1,x)==1				
					out(x_train_idx,new_col_idx) = pred1(1,x);	
					label_train(1,x_train_idx)=orig(1,x);	
					x_train_idx=x_train_idx+1;
									
				else
					%ismember(x,test_idx)				
					out_test(x_test_idx,new_col_idx) = pred1(1,x);
					label_test(1,x_test_idx)=orig(1,x);	
					x_test_idx=x_test_idx+1;
				end 
			end
			new_col_idx=new_col_idx+1;
		end
	end
	%write prediction in a file in column
	disp(size(label_train));
	csvwrite('predictions_all.txt', out)
	csvwrite('predictions_test_all.txt', out_test)
	csvwrite('label_train_all.txt', label_train)
	csvwrite('label_test_all.txt', label_test)
	
	
	
	
