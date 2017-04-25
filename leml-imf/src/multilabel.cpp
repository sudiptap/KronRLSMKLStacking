
#include "multilabel.h"
#include "smat.h"
#include "multiple_linear.h"
#include "bilinear.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


void update_H(smat_t &Y, smat_t&X, double *W, int k, multilabel_parameter *param, double *H){
	double *XW = Malloc(double, X.rows*k);
	smat_x_dmat(X, W, k, XW);
	multiple_linear_problem subprob(&Y, XW, k);
	multiple_linear_parameter subparam(param->Cp, param->Cn,
			param->max_tron_iter, param->max_cg_iter, param->eps, param->verbose);
	switch (param->solver_type) {
		case L2R_BILINEAR_LS:
			{
				multiple_l2r_ls_chol(&subprob, &subparam, H);
				//	multiple_l2r_ls_tron(&subprob, &subparam, H);
				break;
			}
		case L2R_BILINEAR_LR:
			{
				multiple_l2r_lr_tron(&subprob, &subparam, H);
				break;
			}
		case L2R_BILINEAR_SVC:
			{
				multiple_l2r_l2svc_tron(&subprob, &subparam, H);
				break;
			}
		case L2R_BILINEAR_LS_FULL: 
			{
				multiple_l2r_ls_chol_full(&subprob, &subparam, H);
				break;
			}
	}
	free(XW);
	return;
}

void update_W(smat_t &Y, smat_t &X, double *H, int k, multilabel_parameter *param, double *W){
	bilinear_problem subprob = bilinear_problem(&Y, &X, H, k, W);
	bilinear_train(&subprob, param, W);
	return;
}

void update_W_subsample(smat_t &Y, smat_t &X, double *H, int k, multilabel_parameter *param, double *W, int nr_samples = 0) {
	int nr_insts = (int)Y.rows;
	if(nr_samples == 0) nr_samples = (int)(0.01*nr_insts);
	std::vector<int>subset = subsample(nr_samples, nr_insts);
	smat_t subY = Y.row_subset(subset);
	smat_t subX = X.row_subset(subset);
	bilinear_problem subprob(&subY, &subX, H, k, W);
	bilinear_train(&subprob, param, W);
}

static double norm(double *W, size_t size) {
	double ret = 0;
	for(size_t i = 0; i < size; i++)
		ret += W[i]*W[i];
	return sqrt(ret);
}


void multilabel_train(multilabel_problem *prob, multilabel_parameter *param, double *W, double *H){
	int k = param->k;
	smat_t &Y = *(prob->training_set->Y);
	smat_t &X = *(prob->training_set->X);
	long nr_insts = Y.rows;
	double threshold = 0;

	if(param->solver_type == L2R_BILINEAR_LS_FULL)
		threshold = 0.5;

	printf("threshold %.2g\n", threshold);
	bilinear_eval_result eval_result(param->top_p, threshold);


	omp_set_num_threads(param->threads);
	printf("threads %d\n", omp_get_max_threads());

	//printf("|W| (%d %ld) = %.10g |H| (%d %ld)= %.10g\n",k, X.cols, norm(W,X.cols*k), k, Y.cols, norm(H,Y.cols*k));
	printf("|H| (%d %ld)= %.10g\n", k, Y.cols, norm(H,Y.cols*k));

	long *perm = MALLOC(long, nr_insts);
	for(long i = 0; i < nr_insts; i++) 
		perm[i] = i;

	double Wtime = 0, Htime = 0, time_start = 0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		{

			time_start = omp_get_wtime();
			update_W(Y, X, H, k, param, W);
			Wtime += omp_get_wtime() - time_start;

			time_start = omp_get_wtime();
			update_H(Y, X, W, k, param, H);
			Htime += omp_get_wtime() - time_start;


			printf("ML-iter %d W %.5g H %.5g Time %.5g ", 
					iter, Wtime, Htime, Wtime+Htime);

		}

		if(iter >= 1 && param->predict) {
			bool predict = true;
			if(predict) {
				bilinear_predict(prob->test_set, W, H, &eval_result);
				printf("AUC %.4g Micro-F1 %.4g Hamming-Loss %.4g ", 
						eval_result.avg_auc, eval_result.micro_F1, eval_result.hamming_loss);
				for(int p = 0; p < param->top_p; p++) 
					printf(" top-%d %.4g", p+1, eval_result.avg_top_real_acc[p]);
			}
		}
		puts("");
		fflush(stdout);
	}
	free(perm);
} 
