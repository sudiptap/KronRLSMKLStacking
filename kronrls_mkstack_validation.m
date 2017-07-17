function kronrls_mkstack1( K1, K2, y, train_idx, lambda, num_ones, y_test, test_idx, y_orig, fold, validation_idx)
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
    %disp(size(train_idx,1));
	%disp(size(test_idx,1));
	%get the number of kernels
    nA = size(K1,3); 
    nB = size(K2,3); 
	stacking_rows = size(y,1)*size(y,2);
	stacking_cols = nA*nB;
	out = zeros(size(train_idx,1), stacking_cols); %disp(size(out));
	out_test = zeros(size(test_idx,1), stacking_cols); %disp(size(out_test));
	out_validation = zeros(size(validation_idx,1), stacking_cols); %disp(size(out_validation));
	%record all results for further analysis
	%prediction = cell(size(K1,1)*size(K1,2));
	%prediction = zeros(:,:);
	%tot_entires = size(K1,1)*size(K1,2);
	new_col_idx = 1; 
	orig = reshape(y_orig, [1, size(y_orig,1)*size(y_orig,2)]);
	%orig_test = reshape(y_test, [1, size(y_test,1)*size(y_test,2)]);
	label_train = zeros(1,size(train_idx,1));
	label_test = zeros(1,size(test_idx,1));
	label_validation = zeros(1,size(validation_idx,1));
	for i=1:nA
		for j=1:nB
			pred = kronrls(K1(:,:,i),K2(:,:,j),y, lambda);
			
			pred1 = reshape(pred, [1, size(y,1)*size(y,2)]);
			%pred_test = pred_train(test_idx);			
			
			%disp(size(pred1));
			x_train_idx=1; x_test_idx=1; x_validation_idx=1;
			for x=1:stacking_rows
				if ismember(x,train_idx)%orig(1,x)==1				
					out(x_train_idx,new_col_idx) = pred1(1,x);	
					label_train(1,x_train_idx)=orig(1,x);	
					x_train_idx=x_train_idx+1;									
				elseif ismember(x,test_idx)
					%ismember(x,test_idx)	
					out_test(x_test_idx,new_col_idx) = pred1(1,x);
					label_test(1,x_test_idx)=orig(1,x);	
					x_test_idx=x_test_idx+1;
				else
					%ismember(x,test_idx)	
				    out_validation(x_validation_idx,new_col_idx) = pred1(1,x);
					label_validation(1,x_validation_idx)=orig(1,x);	
					x_validation_idx=x_validation_idx+1;					
				end 
			end
			new_col_idx=new_col_idx+1;
		end
	end
	%write prediction in a file in column
	%disp(size(label_train));
	
	mkdir(int2str(fold));
	cd(int2str(fold));
	csvwrite('predictions_all.txt', out)
	csvwrite('predictions_test_all.txt', out_test)
	csvwrite('predictions_validation_all.txt', out_validation)
	csvwrite('label_train_all.txt', label_train)
	csvwrite('label_test_all.txt', label_test)
	csvwrite('label_validation_all.txt', label_validation)
	cd ..;
	
	
	
	