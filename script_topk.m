
nruns = 1;
dataname_set = cell(1,4);
dataname_set{1} = 'nr';
dataname_set{2} = 'gpcr';
dataname_set{3} = 'ic';
dataname_set{4} = 'e';

result_root = 'result_topk';
for i = 1:3
  dataname = dataname_set{i}
  for nfolds = 5:20
    path = strcat(result_root, '/', dataname, '/', num2str(nfolds), '/');
    mkdir(path);
    path
    sample_topk(nfolds, nruns, dataname, path);
  end
end


