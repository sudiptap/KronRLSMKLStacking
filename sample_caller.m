function [] = sample_caller(nfolds, nruns, dataname)
	aupr_list = [];
	for nfolds = 17:20
		disp(nfolds);
		aupr_out = sample(nfolds,1,'ic');
		aupr_list = [aupr_list aupr_out];
	end
	disp(aupr_list);
	%fileID = fopen('kronrls_mkl_output.txt','w');
	dlmwrite('kronrls_mkl_output.txt',aupr_list);
	%fclose(fileID);
end