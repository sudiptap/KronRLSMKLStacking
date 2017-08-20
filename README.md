## Kronrls-stacking
Matlab and python implementation of the Kronecker Regularized Least Squares with multiple kernel stacking algorithm.

## Pre-requisite
Matlab, Python, Sklearn, Numpy

## Usage
Please follow the steps below to run the code:
1. Run sample_kronrlsmkstack_validation.m . This generates predictions for all pariwise kernels for train, test and validation datasets.
2. Run pairwise_kronrls_with_validation.py . This ranks the kernel pairs according to validation AUPR for top-k selection
3. Run stacking_various_algo.py . This will use top-k kernels to produce test AUPR over all 5 folds.

Data matrices can be downloaded from http://www.cin.ufpe.br/~acan/kronrlsmkl/

## To Do:
Please note that the code has a number of hardcoded parameter right now. We are working on the code to remove these parameters to move everything to commandline.

## Please cite:
coming soon

## Contact:
Please contact at sudipto.pathak@gmail.com for questions and queries.
