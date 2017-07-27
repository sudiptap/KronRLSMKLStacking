function [] = sample()
clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1; n_validation_folds=50

dataname = 'nr'; %please use nflods=17 for best result
%dataname = 'gpcr'; %please use nflods=10 for best result
%dataname = 'ic'; %please use nflods=20 for best result
%dataname = 'e';


% load adjacency matrix
[y,l1,l2] = loadtabfile(['data/interactions/' dataname '_admat_dgc.txt']);
run_aupr  = [];
alpha_m = [];

%how many 1s do we have
num_ones = 0;

number_of_rows = size(y,1)
number_of_cols = size(y,2)


for y_row =1:size(y,1)
	for y_col = 1:size(y,2)
		if y(y_row, y_col)==1
			num_ones = num_ones+1;
		end
	end
end
%num_ones = 10;
for run=1:1
    % split folds
    crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
	%extract the validation now
	
    fold_aupr = [];

    for fold=1:nfolds
        fprintf('---------------\nRUN %d - FOLD %d \n', run, fold)
        train_idx = find(crossval_idx~=fold); 
        test_idx  = find(crossval_idx==fold);	
		%validation_idx = find(validation_idx)
		
        y_train1 = y; y_test = y;
        y_train1(test_idx) = 0; %disp(size(train_idx)); disp(size(test_idx));
		
		    %validation block - start
		    
		    %training dataset is extracted
		    %extract the validation dataset now
		    validation_idx_idx = crossvalind('Kfold', length(train_idx), n_validation_folds);
		    train_idx_tmp = train_idx(find(validation_idx_idx~=n_validation_folds)); 
            validation_idx  = train_idx(find(validation_idx_idx==n_validation_folds));
		    train_idx = train_idx_tmp;
		    y_train = y_train1; y_validation = y_train;
            y_train(validation_idx) = 0; %disp(size(train_idx)); disp(size(test_idx));
		    y_validation(train_idx)=0;
		    y_test(train_idx)=0; %y_test(test_idx)=1;
		    y_test(validation_idx)=0;
		    disp(intersect(train_idx,validation_idx));
		    disp(intersect(test_idx,validation_idx));
			disp(intersect(test_idx,train_idx));
			%disp((union(union(train_idx, test_idx), validation_idx)));
			%disp(length(y(:)));
		    %validation_file_name = strcat('example_validation_',int2str(fold),'.txt');		
		    %dlmwrite(validation_file_name,reshape(y_validation, [number_of_rows number_of_cols]), '\t')
		    
		    %validation block - end
		    
		    num_ones = nnz(y_train);
		    %disp(num_ones);
		    y_train_temp = reshape(y_train,[1,size(y_train,1)*size(y_train,2)]);
		    csvwrite('train.csv', y_train_temp);
		    %csvwrite('test.txt', y_train());
        %% load kernels
        k1_paths = {['data/kernels/' dataname '_simmat_proteins_sw-n.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_go.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k3m1.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k3m2.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k4m1.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k4m2.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_spectrum-n-k3.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_spectrum-n-k4.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_ppi.txt'],...
                    };
        K1 = [];
        for i=1:length(k1_paths)
            [mat, labels] = loadtabfile(k1_paths{i});
            mat = process_kernel(mat);
            K1(:,:,i) = mat;
        end
        K1(:,:,i+1) = kernel_gip(y_train,1, 1);

        k2_paths = {['data/kernels/' dataname '_simmat_drugs_simcomp.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_lambda.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_marginalized.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_minmaxTanimoto.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_spectrum.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_tanimoto.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_aers-bit.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_aers-freq.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_sider.txt'],...
                    };
        K2 = [];
        for i=1:length(k2_paths)
            [mat, labels] = loadtabfile(k2_paths{i});
            mat = process_kernel(mat);
            K2(:,:,i) = mat;
        end
        K2(:,:,i+1) = kernel_gip(y_train,2, 1);

        %% perform predictions
        lambda = 1;
        regcoef = 0.25;
        
        %[ y2 , alpha, beta ] = kronrls_mkl( K1, K2, y_train, lambda, regcoef, num_ones );
		%disp(num_ones);
		%[ y2 ] = kronrls_pairwise_mkboost( K1, K2, y_train, lambda, regcoef, 50, true, num_ones );
		%[ y2 ] = kronrls_mkboost( K1, K2, y_train, lambda, regcoef, 50, true, num_ones );
		%write the train and test files now - later will be used by autoencoders
		%disp(reshape(y_train, [number_of_rows number_of_cols]))
		%dlmwrite('example_train.txt',reshape(y_train, [number_of_rows number_of_cols], 'delimiter', '\t'))
		%dlmwrite('example_test.txt',reshape(y_test, [number_of_rows number_of_cols], 'delimiter', '\t'))
		
		
		
		
		
		
		%disp(reshape(y_test, [number_of_rows number_of_cols]))
		
		kronrls_mkstack_validation( K1, K2, y_train, train_idx, lambda, num_ones, y_test, test_idx, y, fold, validation_idx);
		%kronrls_mkstack( K1, K2, y_train, train_idx, lambda, num_ones, y_test, test_idx, y);
        
        % evaluate predictions
        yy=y; 
        yy(yy==0)=-1;%disp(size(yy)); return;
		
        %stats = evaluate_performance(y2(test_idx),yy(test_idx),'classification');

        %fold_aupr = [fold_aupr, stats.aupr];
    end
    
    %run_aupr(run,:)=fold_aupr;
end
%mean(mean(run_aupr,2))
end
