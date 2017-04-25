

#include "mex.h"
#include "../multilabel.h"
#include "../bilinear.h"
#include <omp.h>
#include <cstring>

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL


void exit_with_help()
{
	mexPrintf(
	"Usage: [W H wall_time] = train_ml(Y, X, Yt, Xt [, 'options'])\n"
	"       [W H wall_time] = train_ml(Y, X, Yt, Xt, W0, H0 [, 'options'])\n"
	"       size(W0) = (rank, nr_feats), size(H0) = (rank, nr_labels)\n"
	"options:\n"
	"    -s type : set type of solver (default 10)\n"    
	"    	 0 -- L2R_LS (Squared Loss)\n"    
	"    	 1 -- L2R_LR (Logistic Regression)\n"    
	"    	 2 -- L2R_SVC (Squared Hinge Loss)\n"    
	"    	 10 -- L2R_LS (Squared Loss) Fully observation\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"    
	"    -t max_iter: set the number of iterations (default 10)\n"    
	"    -T max_tron_iter: set the number of iterations used in TRON (default 5)\n"    
	"    -g max_cg_iter: set the number of iterations used in CG (default 20)\n"
	"    -e epsilon : set inner termination criterion epsilon of TRON (default 0.1)\n"     
	"    -P top-p: set top-p accruacy (default 5)\n"
	"    -q show_predict: set top-p accruacy (default 1)\n"
	);
}

multilabel_parameter parse_command_line(int nrhs, const mxArray *prhs[])
{
	multilabel_parameter param;   // default values have been set by the constructor 
	int i, argc = 1;
	int option_pos = -1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	param.predict = true;
	param.top_p = 5;
	param.solver_type = L2R_BILINEAR_LS_FULL; 

	if(nrhs == 4)
		return param;
	if(nrhs == 5)
		option_pos = 4;
	if(nrhs == 7)
		option_pos = 6;

	// put options in argv[]
	if(option_pos>0)
	{
		mxGetString(prhs[option_pos], cmd,  mxGetN(prhs[option_pos]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.Cp = 1/(atof(argv[i]));
				param.Cn = param.Cp;
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.max_tron_iter = atoi(argv[i]);
				break;

			case 'g':
				param.max_cg_iter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'r':
				param.lrate = atof(argv[i]);
				break;

			case 'P':
				param.predict = true;
				param.top_p = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			default:
				mexPrintf("unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (nrhs > 5) {
		if(mxGetM(prhs[4]) != mxGetM(prhs[5])) 
			mexPrintf("Dimensions of W and H do not match!\n");
		if(param.k != (int)mxGetM(prhs[4])) {
			param.k = (int)mxGetM(prhs[4]);
			mexPrintf("Change param.k to %d.\n", param.k);
		}
	}

	return param;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int transpose(const mxArray *M, mxArray **Mt) {
	mxArray *prhs[1] = {const_cast<mxArray *>(M)}, *plhs[1];
	if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
	{
		mexPrintf("Error: cannot transpose training instance matrix\n");
		return -1;
	}
	*Mt = plhs[0];
	return 0;
}

// convert matlab sparse matrix to C smat fmt

class mxSparse_iterator_t: public entry_iterator_t {
	private:
		mxArray *Mt;
		mwIndex *ir_t, *jc_t;
		double *v_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		mxSparse_iterator_t(const mxArray *M){
			rows = mxGetM(M); cols = mxGetN(M);
			nnz = *(mxGetJc(M) + cols); 
			transpose(M, &Mt);
			ir_t = mxGetIr(Mt); jc_t = mxGetJc(Mt); v_t = mxGetPr(Mt);
			cur_idx = cur_row = 0;
		}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= jc_t[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, ir_t[cur_idx], v_t[cur_idx]);
			cur_idx++;
			return ret;
		}
		~mxSparse_iterator_t(){
			mxDestroyArray(Mt);
		}

};
smat_t mxSparse_to_smat(const mxArray *M, smat_t &R) {
	long rows = mxGetM(M), cols = mxGetN(M), nnz = *(mxGetJc(M) + cols); 
	mxSparse_iterator_t entry_it(M);
	R.load_from_iterator(rows, cols, nnz, &entry_it);
	return R;
}

/*
blocks_t mxSparse_to_blocks(const mxArray *M, int num_blocks, blocks_t &R) {
	R = blocks_t(num_blocks);
	unsigned long rows, cols, nnz;
	mwIndex *ir, *jc;
	double *v;
	ir = mxGetIr(M); jc = mxGetJc(M); v = mxGetPr(M);
	rows = mxGetM(M); cols = mxGetN(M); nnz = jc[cols];
	R.from_matlab(rows, cols, nnz);
	for(unsigned long c = 0; c < cols; c++) {
		for(unsigned long idx = jc[c]; idx < jc[c+1]; ++idx){
			R.insert_rate(idx, rate_t(ir[idx], c, v[idx]));
			++R.nnz_row[ir[idx]];
			++R.nnz_col[c];
		}
	}
	R.compressed_space(); // Need to call sort later.
	sort(R.allrates.begin(), R.allrates.end(), RateComp(&R));
	return R;
}

// convert matab dense matrix to column fmt
int mxDense_to_matCol(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(cols, vec_t(rows,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[c][r] = val[idx++];
	return 0;
}

int matCol_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long cols = M.size(), rows = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matCol_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[c][r];
	return 0;
}

// convert matab dense matrix to row fmt
int mxDense_to_matRow(const mxArray *mxM, mat_t &M) {
	unsigned long rows = mxGetM(mxM), cols = mxGetN(mxM);
	double *val = mxGetPr(mxM);
	M = mat_t(rows, vec_t(cols,0));
	for(unsigned long c = 0, idx = 0; c < cols; ++c) 
		for(unsigned long r = 0; r < rows; ++r)
			M[r][c] = val[idx++];
	return 0;
}

int matRow_to_mxDense(const mat_t &M, mxArray *mxM) {
	unsigned long rows = M.size(), cols = M[0].size();
	double *val = mxGetPr(mxM);
	if(cols != mxGetN(mxM) || rows != mxGetM(mxM)) {
		mexPrintf("matRow_to_mxDense fails (dimensions do not match)\n");
		return -1;
	}

	for(unsigned long c = 0, idx = 0; c < cols; ++c)
		for(unsigned long r = 0; r < rows; r++) 
			val[idx++] = M[r][c];
	return 0;
}
*/
double* rand_init(double *M, int m, int n) {
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			M[i*n+j] = drand48();
	return M;
}

mxArray* creat_speye(int n){
	mxArray *ret = mxCreateSparse(n, n, n, mxREAL);
	double *val = mxGetPr(ret);
	mwIndex *ir = mxGetIr(ret);
	mwIndex *jc = mxGetJc(ret);
	for(int i = 0; i < n; i ++) {
		ir[i] = (mwIndex) i;
		jc[i] = (mwIndex) i;
		val[i]= (double) 1.0;
	}
	jc[n] = (mwIndex) n;
	return ret;
}

int run_multilabel_train(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], multilabel_parameter &param) {
	int k = param.k;
	const mxArray *mxY=prhs[0], *mxX=prhs[1], *mxYt=prhs[2], *mxXt=prhs[3], *mxH, *mxW;
	smat_t Y,Yt,X,Xt;
	mxSparse_to_smat(mxY, Y); mxSparse_to_smat(mxX, X);
	mxSparse_to_smat(mxYt, Yt); mxSparse_to_smat(mxXt, Xt);

	double *W, *H; 
	W = mxGetPr(plhs[0] = mxCreateDoubleMatrix(param.k,X.cols,mxREAL));
	H = mxGetPr(plhs[1] = mxCreateDoubleMatrix(param.k,Y.cols,mxREAL));

	bilinear_problem training_set(&Y, &X, H, k);
	bilinear_problem test_set(&Yt, &Xt, H, k);


	if (nrhs > 5) {
		mxW=prhs[4]; mxH=prhs[5];
		training_set.W = (double*)memcpy(W, mxGetPr(mxW), sizeof(double)*k*X.cols);
		training_set.H = (double*)memcpy(H, mxGetPr(mxH), sizeof(double)*k*Y.cols);
	} else {
		srand48(0UL);
		training_set.W = rand_init(W, (int)X.cols, k);
		training_set.H = rand_init(H, (int)Y.cols, k);
	}

	multilabel_problem prob(&training_set, &test_set);
	omp_set_num_threads(param.threads);
	double start = omp_get_wtime();
	multilabel_train(&prob, &param, W, H);
	double *wtime = mxGetPr(plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL));
	*wtime = omp_get_wtime() - start;
}

bool isDoubleSparse(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(!mxIsSparse(mxM)) {
			mexPrintf("matrix must be sparse; "
					"use sparse(matrix) first\n");
			return false;
		}
		return true;
}
bool isDoubleDense(const mxArray *mxM) {
		if(!mxIsDouble(mxM)) {
			mexPrintf("Error: matrix must be double\n");
			return false;
		}

		if(mxIsSparse(mxM)) {
			mexPrintf("matrix must be dense; "
					"use full(matrix) first\n");
			return false;
		}
		return true;
}
// Interface function of matlab
// now assume prhs[0]: A, prhs[1]: W, prhs[0]
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	multilabel_parameter param;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);

	// Transform the input Matrix to libsvm format
	if(4 <= nrhs && nrhs <= 7)
	{
		if (isDoubleSparse(prhs[0])==false
				|| isDoubleSparse(prhs[1])==false
				|| isDoubleSparse(prhs[2])==false
				||isDoubleSparse(prhs[3])==false) {
			fake_answer(plhs);
			return;
		}
		if(nrhs>5 && (isDoubleDense(prhs[4])==false
					|| isDoubleDense(prhs[5])==false)) {
			fake_answer(plhs);
			return;
		}
		if(!mxIsDouble(prhs[0])) {
			mexPrintf("Error: matrix must be double\n");
			fake_answer(plhs);
			return;
		}

		if(!mxIsSparse(prhs[0])) {
			mexPrintf("Training_matrix must be sparse; "
					"use sparse(Training_matrix) first\n");
			fake_answer(plhs);
			return;
		}

		param = parse_command_line(nrhs, prhs);
		run_multilabel_train(nlhs, plhs, nrhs, prhs, param);
	} else {
		exit_with_help();
		fake_answer(plhs);
	}

}


