
#include "mf.h"
#include "smat.h"
#include "dmat.h"
#include "multiple_linear.h"
#include "bilinear.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void update_WW(mf_problem *prob, mf_parameter *param, double *nk_buf, double *W, double *H){
	dmat_x_dmat_rowmajor(1.0, prob->XX2, false, H, false, 0.0, nk_buf, (size_t)prob->n, (size_t)prob->k, (size_t)prob->f2);
	bilinear_problem subprob = bilinear_problem(prob->Y, prob->XX1, prob->f1, nk_buf, (int)param->k, W);
	bilinear_train(&subprob, param, W);
	return;
}

void update_HH(mf_problem *prob, mf_parameter *param, double *mk_buf, double *W, double *H){
	dmat_x_dmat_rowmajor(1.0, prob->XX1, false, W, false, 0.0, mk_buf, (size_t)prob->m, (size_t)prob->k, (size_t)prob->f1);
	smat_t Yt; Yt = prob->Y->transpose();
	bilinear_problem subprob = bilinear_problem(&Yt, prob->XX2, prob->f2, mk_buf, (int)param->k, H);
	bilinear_train(&subprob, param, H);
	return;
}

void update_W(mf_problem *prob, mf_parameter *param, double *nk_buf, double *W, double *H){
	if (prob->isSparseX2) smat_x_dmat(*(prob->X2), H, param->k, nk_buf);
	else dmat_x_dmat_rowmajor(1.0, prob->XX2, false, H, false, 0.0, nk_buf, (size_t)prob->n, (size_t)prob->k, (size_t)prob->f2);
	if (prob->isSparseX1) {
		bilinear_problem subprob = bilinear_problem(prob->Y, prob->X1, nk_buf, (int)param->k, W);
		bilinear_train(&subprob, param, W);
	}	
	else {
		bilinear_problem subprob = bilinear_problem(prob->Y, prob->XX1, prob->f1, nk_buf, (int)param->k, W);
		bilinear_train(&subprob, param, W);
	}
	return;
}

void update_H(mf_problem *prob, mf_parameter *param, double *mk_buf, double *W, double *H){
	if (prob->isSparseX1) smat_x_dmat(*(prob->X1), W, param->k, mk_buf);
	else dmat_x_dmat_rowmajor(1.0, prob->XX1, false, W, false, 0.0, mk_buf, (size_t)prob->m, (size_t)prob->k, (size_t)prob->f1);
	smat_t Yt; Yt = prob->Y->transpose();
	if (prob->isSparseX2) {
		bilinear_problem subprob = bilinear_problem(&Yt, prob->X2, mk_buf, (int)param->k, H);
		bilinear_train(&subprob, param, H);
	}
	else {
		bilinear_problem subprob = bilinear_problem(&Yt, prob->XX2, prob->f2, mk_buf, (int)param->k, H);
		bilinear_train(&subprob, param, H);
	}
	return;
}


static double norm(double *W, size_t size) {
	double ret = 0;
	for(size_t i = 0; i < size; i++)
		ret += W[i]*W[i];
	return sqrt(ret);
}

void mf_train(mf_problem *prob, mf_parameter *param, double *W, double *H){
	long m = prob->Y->rows, n = prob->Y->cols;
	long f1 = prob->f1, f2 = prob->f2;
	int k = param->k;
	double threshold = 0;

	double *mk_buf = MALLOC(double, m*k);
	double *nk_buf = MALLOC(double, n*k);

	if(param->solver_type == L2R_BILINEAR_LS_FULL)
		threshold = 0.5;
	printf("threshold %.2g\n", threshold);

	omp_set_num_threads(param->threads);
	printf("threads %d\n", omp_get_max_threads());

	printf("|W| (%d %ld)= %.10g\n", k, f1, norm(W,f1*k));
	printf("|H| (%d %ld)= %.10g\n", k, f2, norm(H,f2*k));

	double Wtime = 0, Htime = 0, time_start = 0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		if(param->solver_type <= L2R_BILINEAR_LS_FULL) {
			time_start = omp_get_wtime();
			update_W(prob, param, nk_buf, W, H);
			Wtime += omp_get_wtime() - time_start;

			time_start = omp_get_wtime();
			update_H(prob, param, mk_buf, W, H);
			Htime += omp_get_wtime() - time_start;

			printf("ML-iter %d W %.5g H %.5g Time %.5g ", 
					iter, Wtime, Htime, Wtime+Htime);
		}
		puts("");
		fflush(stdout);
	}
	free(nk_buf);
	free(mk_buf);
} 



void mf_train_dense(mf_problem *prob, mf_parameter *param, double *W, double *H){
	long m = prob->Y->rows, n = prob->Y->cols;
	long f1 = prob->f1, f2 = prob->f2;
	int k = param->k;
	double threshold = 0;

	double *mk_buf = MALLOC(double, m*k);
	double *nk_buf = MALLOC(double, n*k);

	if(param->solver_type == L2R_BILINEAR_LS_FULL)
		threshold = 0.5;
	printf("threshold %.2g\n", threshold);

	omp_set_num_threads(param->threads);
	printf("threads %d\n", omp_get_max_threads());

	printf("|W| (%d %ld)= %.10g\n", k, f1, norm(W,f1*k));
	printf("|H| (%d %ld)= %.10g\n", k, f2, norm(H,f2*k));

	double Wtime = 0, Htime = 0, time_start = 0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		if(param->solver_type <= L2R_BILINEAR_LS_FULL) {
			time_start = omp_get_wtime();
			update_WW(prob, param, nk_buf, W, H);
			Wtime += omp_get_wtime() - time_start;

			time_start = omp_get_wtime();
			update_HH(prob, param, mk_buf, W, H);
			Htime += omp_get_wtime() - time_start;

			printf("ML-iter %d W %.5g H %.5g Time %.5g ", 
					iter, Wtime, Htime, Wtime+Htime);
		}
		puts("");
		fflush(stdout);
	}
	free(nk_buf);
	free(mk_buf);
} 
