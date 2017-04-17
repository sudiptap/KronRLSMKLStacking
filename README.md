# KronRLSMKLStacking
Kronecker Regularized Least Squares with Multiple Kernels Learning through Stacking

Steps to run :
------------------
1. Open Matlab and call sample_kronrlsmkstack.m
2. Please look at the sample_kronrlsmkstack.m code to change datasets. Please use nfolds-value accordingly for best AUPR. nfolds values are commented in the code.
3. sample_kronrlsmkstack produces output files to be used for AUPR calculation. These files are the following: predictions_all.txt, predictions_test_all.txt, label_test_all.txt and label_train_all.txt. 
4. Run stacking1.py to plot and generate AUPR. Please make sure you have the above mentioned files in current directory.
