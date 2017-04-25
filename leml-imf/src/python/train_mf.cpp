#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
#include "../mf.h"


static PyObject *s_data = PyString_FromString("data");
static PyObject *s_indptr = PyString_FromString("indptr");
static PyObject *s_indices = PyString_FromString("indices");
static PyObject *s_dtype = PyString_FromString("dtype");
static PyObject *s_name = PyString_FromString("name");
static PyObject *s_shape = PyString_FromString("shape");
static bool decref = true;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

static bool check_typename(PyObject *o, char *name) {
    return strcmp(o->ob_type->tp_name, name) == 0;
}

static int sparse_get_rows(PyObject *s) {
    int rows = -1;
    //printf("get_rows Refcount S %d\n", s->ob_refcnt);
    //printf("get_rows Refcount s_shape %d\n", s_shape->ob_refcnt);
    PyObject *shape = PyObject_GetAttr(s, s_shape); // new reference
    //printf("VVV %d, %d\n", s_shape->ob_refcnt, shape->ob_refcnt);
    PyObject *py_rows = PyTuple_GetItem(shape, 0); // borrowed reference
    //printf("XXX %d, %d, %d\n", s_shape->ob_refcnt, shape->ob_refcnt, py_rows->ob_refcnt);

    rows = (int)PyInt_AsLong(py_rows);
    //printf("YYY %d, %d, %d\n", s_shape->ob_refcnt, shape->ob_refcnt, py_rows->ob_refcnt);

    if (decref) {Py_DECREF(shape);}
    //printf("leaving get_rows Refcount s_shape %d, %d, %d\n", s_shape->ob_refcnt, shape->ob_refcnt, py_rows->ob_refcnt);
    return rows;
}

static int sparse_get_cols(PyObject *s) {
    PyObject *shape = PyObject_GetAttr(s, s_shape); // new reference
    PyObject *py_cols = PyTuple_GetItem(shape, 1); // borrowed reference
    int result = (int)PyInt_AsLong(py_cols);
    if (decref) {Py_DECREF(shape);}
    return result;
}

static int sparse_get_data_numpy_type(PyObject *s) {
    // static PyObject *s_data = PyString_FromString("data");
    PyArrayObject *data = (PyArrayObject*)PyObject_GetAttr(s, s_data); // new reference
    int r = PyArray_TYPE(data);
    if (decref) {Py_DECREF(data);}
    return r;
}

static unsigned int *sparse_get_indptr(PyObject *A) {
    PyArrayObject *a_indptr;
    a_indptr = (PyArrayObject*)PyObject_GetAttr(A, s_indptr); // new refernce
    if (decref) {Py_DECREF(a_indptr);}
    return (unsigned int*)PyArray_DATA(a_indptr);
}

static unsigned int *sparse_get_indices(PyObject *A) {
    PyArrayObject *a_indices;
    a_indices = (PyArrayObject*)PyObject_GetAttr(A, s_indices); // new reference
    if (decref) {Py_DECREF(a_indices);}
    return (unsigned int*)PyArray_DATA(a_indices);
}

static double *sparse_get_data(PyObject *A) {
    PyArrayObject *a_data;
    a_data = (PyArrayObject*)PyObject_GetAttr(A, s_data); // new reference
    if (decref) {Py_DECREF(a_data);}
    return (double*)PyArray_DATA(a_data);
}

static int sparse_get_nnz(PyObject *A) {
    PyArrayObject *a_data = (PyArrayObject*)PyObject_GetAttr(A, s_data); // new reference
    int result = (int)PyArray_DIM(a_data, 0);
    if (decref) {Py_DECREF(a_data);}
    return result;
}

class scipy_sparse_iterator_t : public entry_iterator_t {
    private:
        double *data;
        unsigned int *indptr, *indices;

    public:
        size_t rows, cols, cur_idx, cur_col_idx;

        scipy_sparse_iterator_t (PyObject *A){
            // this->validate(A);
            this->nnz = sparse_get_nnz(A);
            this->indptr = sparse_get_indptr(A);
            this->indices = sparse_get_indices(A);
            this->data = sparse_get_data(A);
            this->rows = sparse_get_rows(A);
            this->cols = sparse_get_cols(A);
            this->cur_idx = 0;
            this->cur_col_idx = 0;
        }

        rate_t next() {
            int row = -1, col = -1;
            double val = 0;

            if (this->cur_idx >= this->nnz) {
                printf("Now you've gone too far\n");
                throw "iterated too far!";
            }

            // set the column
            while (this->cur_idx >= this->indptr[this->cur_col_idx+1]) {
                this->cur_col_idx++;
            }

            col = (int)cur_col_idx;
            row = this->indices[cur_idx];
            val = this->data[cur_idx];

            rate_t result(row, col, val);
            cur_idx++;
            return result;
        }

        ~scipy_sparse_iterator_t(){
            // printf("destroying sparse iterator\n");
        }
};

smat_t scipy_sparse_to_smat(PyObject *A, smat_t &R) {
    scipy_sparse_iterator_t ssit(A);
    R.load_from_iterator(ssit.rows, ssit.cols, ssit.nnz, &ssit);
        return R;
}


static PyObject*
train_mf_entry (PyObject *dummy, PyObject *args, PyObject *kwds)
{
    PyObject *Y=NULL, *X1=NULL, *X2=NULL;
    PyArrayObject *W=NULL, *H=NULL;
    smat_t smat_Y, smat_X1, smat_X2;
    double *Wdata, *Hdata;
	double lamb = 0.1;
    int k = 10;
	int solver_type = 0;
    int threads = 4;
    int maxiter = 10;
    static char *kwlist[] = {(char*)"Y", (char *)"X1", (char *)"X2", (char *)"W", (char *)"H",
							(char *)"k", (char *)"lamb", (char *)"solver_type",
							(char *)"maxiter", (char *)"threads", NULL};



    // unpack the arguments with PyArg_ParseTupleAndKeywords()
    //
    // The arguments are borrowed refereneces. Sectoin 1.10.2 of Python C API:
    // "
    // When a C function is called from Python, it borrows references to its 
    // arguments from the caller. The caller owns a reference to the object, so
    // the borrowed reference’s lifetime is guaranteed until the function 
    // returns. Only when such a borrowed reference must be stored or passed 
    // on, it must be turned into an owned reference by calling Py_INCREF().
    // "
    //
    //

    // mf_train(Y, X1, X2, W, H, k)
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOi|diii", kwlist,
                                    &Y, &X1, &X2, &W, &H, &k,
									&lamb, &solver_type, &maxiter, &threads)) {
        printf("Doh!");
        return NULL;
    }

    int d1, d2; // number of user and item features respectively
    int n1, n2; // number of users and items respectively
    printf("maxiter %d\n", maxiter);

    do {
        if (!PyArray_Check(W) || !PyArray_Check(H)) {
            failmsg("train_mf_entry: W and H must be numpy arrays");
            break;
        }

        if (PyArray_NDIM(W) != 2 || PyArray_NDIM(H) != 2) {
            failmsg("train_mf_entry: W and H must have two dimensions");
            break;
        }

        if (!PyArray_DIM(W, 1) == k || PyArray_DIM(H, 1) != k) {
            failmsg("train_mf_entry: W and H must have k cols");
            break;
        }

        if (PyArray_TYPE(W) != NPY_DOUBLE || PyArray_TYPE(H) != NPY_DOUBLE ) {
            failmsg("train_mf_entry: W and H must have dtype double");
            break;
        }

        if (!check_typename(Y, (char *)"csc_matrix") ||
                !check_typename(X1, (char *)"csc_matrix") ||
                !check_typename(X2, (char *)"csc_matrix")) {
            failmsg("train_mf_entry: Y, X1 and X2 must have type csc_matrix");
            break;
        }

        d1 = (int)PyArray_DIM(W, 0);
        d2 = (int)PyArray_DIM(H, 0);
        n1 = sparse_get_rows(X1);
        n2 = sparse_get_rows(X2);

        if (n1 != sparse_get_rows(Y) || n2 != sparse_get_cols(Y)) {
            failmsg("Y shape is incompatible with X1 nrows or X2 ncols");
            break;
        }

        if (d1 != sparse_get_cols(X1)) {
            failmsg("train_mf_entry: X1 (%d cols) and W (%d rows) have incompatible shapes", 
                    sparse_get_cols(X1), d1);
            break;
        }

        if (d2 != sparse_get_cols(X2)) {
            failmsg("train_mf_entry: X2 (%d cols) and H (%d rows) have incompatible shapes", 
                    sparse_get_cols(X2), d2);
            break;
        }

        if (sparse_get_data_numpy_type(Y) != NPY_DOUBLE
                || sparse_get_data_numpy_type(X1) != NPY_DOUBLE
                || sparse_get_data_numpy_type(X2) != NPY_DOUBLE) {
            failmsg("train_mf_entry: data in Y, X1 and X2 must be double");
            break;
        }

        scipy_sparse_to_smat(Y, smat_Y);
        scipy_sparse_to_smat(X1, smat_X1);
        scipy_sparse_to_smat(X2, smat_X2);
        Wdata = (double*)PyArray_DATA(W);
        Hdata = (double*)PyArray_DATA(H);

        mf_problem mf_prob(&smat_Y, &smat_X1, &smat_X2, k);
        mf_parameter mf_param;
		mf_param.solver_type = solver_type;
        mf_param.k = k;
		mf_param.Cp = 1/lamb;
		mf_param.Cn = mf_param.Cp;
		printf("Cp => %g, Cn => %g\n", mf_param.Cp, mf_param.Cn);
        mf_param.threads = threads;
        mf_param.maxiter = maxiter;
        mf_train(&mf_prob, &mf_param, Wdata, Hdata);

        Py_INCREF(Py_None);
        return Py_None;
    } while(0);

    // FAIL!
    printf("somekind of failure\n");

    return NULL;
}

/*
* Bind Python function names to our C functions
*/
static PyMethodDef myModule_methods[] = {
    {"train_mf", (PyCFunction)train_mf_entry, METH_VARARGS | METH_KEYWORDS},
    {NULL, NULL}
};

/*
 * Python calls this to let us initialize our module
*/
PyMODINIT_FUNC init__train_mf(void)
{
    printf("inittrain_mf()\n");
    (void) Py_InitModule("__train_mf", myModule_methods);

    // Need these for numpy arrays
    import_array();
    // import_ufunc();
}


