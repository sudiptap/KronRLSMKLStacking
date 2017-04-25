#ifndef MF_H
#define MF_H

#include "bilinear.h"

class mf_problem {
	public:
		smat_t *Y;  // m*n sparse matrix
		smat_t *X1; // m*f1 sparse matrix
		smat_t *X2; // n*f2 sparse matrix
		double *XX1; // m*f1 dense matrix
		double *XX2; // n*f2 dense matrix
		double *W;  // f1*k array row major W(t,s)=W[k*t+s]
		double *H;  // f2*k array row major H(t,s)=H[k*t+s]
		long m, n;  // dimension of Y
		long f1, f2; // #features of X1 and X2
		long k;
		bool isSparseX1;
		bool isSparseX2;
		mf_problem(){};
		mf_problem(smat_t *Y, smat_t *X1, smat_t *X2, int k, double *W = NULL, double *H = NULL){
			this->Y = Y;
			this->X1 = X1;
			this->X2 = X2;
			this->XX1 = NULL;
			this->XX2 = NULL;
			this->m = Y->rows;
			this->n = Y->cols;
			this->f1 = X1->cols;
			this->f2 = X2->cols;
			this->k = k;
			this->W = W;
			this->H = H;
			this->isSparseX1 = true;
			this->isSparseX2 = true;
		}
		mf_problem(smat_t *Y, double *XX1, double *XX2, size_t f1, size_t f2, int k, double *W = NULL, double *H = NULL){
			this->Y = Y;
			this->X1 = NULL;
			this->X2 = NULL;
			this->XX1 = XX1;
			this->XX2 = XX2;
			this->m = Y->rows;
			this->n = Y->cols;
			this->f1 = f1; // X1->cols;
			this->f2 = f2; // X2->cols;
			this->k = k;
			this->W = W;
			this->H = H;
			this->isSparseX1 = false;
			this->isSparseX2 = false;
		}
		mf_problem(smat_t *Y, smat_t *X1, double *XX1, smat_t *X2, double *XX2, size_t f1, size_t f2, int k, double *W = NULL, double *H = NULL){
			this->Y = Y;
			this->X1 = X1;
			this->X2 = X2;
			this->XX1 = XX1;
			this->XX2 = XX2;
			this->m = Y->rows;
			this->n = Y->cols;
			this->f1 = f1; // X1->cols;
			this->f2 = f2; // X2->cols;
			this->k = k;
			this->W = W;
			this->H = H;
			this->isSparseX1 = XX1==NULL? true:false;
			this->isSparseX2 = XX2==NULL? true:false;
		}
};

class mf_parameter: public bilinear_parameter{ 
	public:
		int maxiter;
		int top_p;
		int k;
		int threads;
		bool predict;
		// Parameters for Wsabie
		double lrate; // learning rate for wsabie
		mf_parameter() {
			bilinear_parameter();
			maxiter = 10; 
			top_p = 20; 
			k = 10;
			threads = 8;
			lrate = 0.01;
			predict = true;
		}
};

#ifdef __cplusplus
extern "C" {
#endif

void mf_train(mf_problem *prob, mf_parameter *param, double *W, double *H);
void mf_train_dense(mf_problem *prob, mf_parameter *param, double *W, double *H);

#ifdef __cplusplus
}
#endif


#endif
