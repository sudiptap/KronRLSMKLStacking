## Kronrls-stacking : Ensemble Learning Algorithm for Drug-Target Interaction Prediction
Predicting drug-target interaction through simulation is an immensely important problem. It has huge impact in drug discovery in pharmaceutical industry. FDA reports that it takes close to five billion dollars to introduce a new drug to the market. A slight improvement in accuracy of prediction in the domain may save thousands if not millions of dollars in investment, there by lowering down the cost of production and making drugs cheaper to its consumers. We proposed a new algorithm to combine multiple heterogeneous information for identification of new interactions between drugs and targets. Empirical results on four data sets show that the proposed strategy has comparable or better prediction accuracy than contemporary methods.

## Pre-requisite
Matlab, Python, Sklearn, Numpy

## Usage
Please follow the steps below to run the code:
1. Run sample_kronrlsmkstack_validation.m . This generates predictions for all pariwise kernels for train, test and validation datasets.
2. Run pairwise_kronrls_xc.py . This ranks the kernel pairs according to validation AUPR for top-k selection
3. Run stacking_various_algo.py . This will use top-k kernels to produce test AUPR over all 5 folds.

Data matrices can be downloaded from http://www.cin.ufpe.br/~acan/kronrlsmkl/

## To Do:
Please note that the code has a number of hardcoded parameter right now. We are working on the code to remove these parameters to move everything to commandline.

## Please cite:
coming soon

## Contact:
Please contact at sudipto.pathak@gmail.com for questions and queries.
