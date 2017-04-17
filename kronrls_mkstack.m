function kronrls_mkstack( K1, K2, y, lambda, num_ones, y_test, test_idx)
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
    disp(size(test_idx));
	%get the number of kernels
    nA = size(K1,3); 
    nB = size(K2,3); 
	stacking_rows = num_ones;%size(y,1)*size(y,2);
	stacking_cols = nA*nB;
	out = zeros(stacking_rows, stacking_cols);
	out_test = zeros(nnz(y_test), stacking_cols);
	%record all results for further analysis
	%prediction = cell(size(K1,1)*size(K1,2));
	%prediction = zeros(:,:);
	%tot_entires = size(K1,1)*size(K1,2);
	new_col_idx = 1; 
	orig = reshape(y, [1, size(y,1)*size(y,2)]);
	orig_test = reshape(y_test, [1, size(y_test,1)*size(y_test,2)]);
	for i=1:nA
		for j=1:nB
			pred = kronrls(K1(:,:,i),K2(:,:,j),y, lambda);
			
			pred_train = reshape(pred, [1, size(pred,1)*size(pred,2)]);
			pred_test = pred_train(test_idx);			
			
			%disp(size(pred1));
			for x=1:stacking_rows
				if orig(1,x)==1				
					out(x,new_col_idx) = pred_train(1,x);
				end
				if orig_test(1,x)==1				
					out_test(x,new_col_idx) = pred_test(1,x);
				end 
			end
			new_col_idx=new_col_idx+1;
		end
	end
	%write prediction in a file in column
	disp(size(out_test));
	csvwrite('predictions_one_class.txt', out)
	csvwrite('predictions_test_one_class.txt', out_test)
	
	
	
	