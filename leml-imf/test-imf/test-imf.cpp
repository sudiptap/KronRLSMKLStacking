
#include <stddef.h>
#include "smat.h"
#include "dmat.h"
#include "mf.h"
#include "bilinear.h"

std::random_device rd;
std::mt19937 e2(rd());
std::normal_distribution<> norm_dist(0,1);

// generate Gaussian random matrix
double* randn_init(double *M, int m, int n) {
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			M[i*n+j] = norm_dist(e2);
	return M;
}

// random test for IMC
void random_test(int k, int d1, int d2, int n1, int n2, int m) {

	smat_t A, Xs, Ys;

	e2.seed(1); // srand48(0UL);

	// set parameters for IMC
	mf_parameter param;
	param.predict = false;
	param.k = k;			// rank
	param.threads = 1;		// number of threads
	param.maxiter = 10;		// number of iterations
	param.Cp = 1/(1e-6);	// lambda
	param.Cn = param.Cp;

	// ground truth Z_t = W*H'
	double *W = MALLOC(double, d1*k);
	double *H = MALLOC(double, d2*k);
	double *Z_t = MALLOC(double, d1*d2);
	randn_init(W, d1, k);
	randn_init(H, d2, k);
	dmat_x_dmat(1.0, W, true, H, false, 0.0, Z_t, d1, d2, k);

	// generate random features X and Y
	double *X = MALLOC(double, d1*n1);
	double *Y = MALLOC(double, d2*n2);
	randn_init(X, d1, n1);
	randn_init(Y, d2, n2);

	// construct A with randomly sampled observations from X*Z*Y'
	unsigned *rowList = MALLOC(unsigned, m);
	unsigned *colList = MALLOC(unsigned, m);
	double *valList = MALLOC(double, m);
	double *s = MALLOC(double, d1);
	std::uniform_int_distribution<> udist1(0,n1-1);
	std::uniform_int_distribution<> udist2(0,n2-1);
	for (int i = 0; i < m; i++) {
		rowList[i] = udist1(e2); // uniform sampling
		colList[i] = udist2(e2); //
		dmat_x_vec(1.0, Z_t, false, &(Y[colList[i]*d2]), 0.0, s, d1, d2);
		valList[i] = do_dot_product(&(X[rowList[i]*d1]), s, d1);
	}
	free(s);

	// convert A to sparse matrix
	A.load_from_array(n1, n2, m, rowList, colList, valList);
	free(rowList); free(colList); free(valList);


	// randomly initialize W and H
	randn_init(W, d1, k);
	randn_init(H, d2, k);

	// set mf problem
	mf_problem prob(&A, &Xs, X, &Ys, Y, d1, d2, k, W, H);

	// run IMC
	double wtime = omp_get_wtime();
	mf_train(&prob, &param, W, H);
	wtime = omp_get_wtime() - wtime;
	
	// computed Z
	double *Z = MALLOC(double, d1*d2);
	dmat_x_dmat(1.0, W, true, H, false, 0.0, Z, d1, d2, k);

	// compute relative error
	do_axpy(-1.0, Z_t, Z, d1*d2);
	double relerr = do_dot_product(Z, Z, d1*d2) / do_dot_product(Z_t, Z_t, d1*d2);

	printf("RelErr = %e  Time = %.4f sec\n",relerr,wtime);
	
	free(Z_t); free(Z); free(W); free(H);
	free(X); free(Y);
}


void print_help() {
	printf(
	"Usage: test-imc k d1 d2 n1 n2 m\n"
	"	k:  rank (default 5)\n"
	"	d1: number of row features (default 50)\n"
	"	d2: number of col features (default 50)\n"
	"	n1: number of rows (default 1000)\n"
	"	n2: number of cols (default 1000)\n"
	"	m:  number of non-zeros in A (default 1000)\n"
	);
}


int main(int argc, char *argv[]) {

	int k = 5, d1 = 50, d2 = 50, n1 = 1000, n2 = 1000, m = 1000;

	// parse options
	if (argc == 7) {
		k = atoi(argv[1]);
		d1 = atoi(argv[2]);
		d2 = atoi(argv[3]);
		n1 = atoi(argv[4]);
		n2 = atoi(argv[5]);
		m = atoi(argv[6]);
	}
	else {
		print_help();
		if (argc == 1)
			printf("Running with default options.\n");
		else
			exit(1);
	}

	// do random test
	random_test(k, d1, d2, n1, n2, m);
	
	
	return 0;
}


