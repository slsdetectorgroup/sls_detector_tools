#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "TH1D.h"
#include "TH2D.h"
#include "TROOT.h"
#include "TF1.h"
#include "TPython.h"

// #include "TPyReturn.h"
// #include "CPyCppyy/API.h"

#include <uuid/uuid.h>

#include "fit_tgraph.h"
#include <Python.h>
#include <iostream>
#include <numpy/arrayobject.h>

/* Docstrings */
static char module_docstring[] =
    "sls_cmodule provies access to compiled c++ and ROOT functions supporting "
    "detector calibration. The library uses numpy arrays as input and return values.";

// TODO! update!
static char load_docstring[] = "Loading binary values from a file";

static char fit_docstring[] = "Fit scurve to all pixels";

PyDoc_STRVAR(find_trimbits_doc, "find_trimbits(data, x, target, par)\n"
                                "--\n"
                                "\n"
                                "Fit all pixels in the data 3d array to extract trimbits.\n\n"
                                "Parameters\n"
                                "----------\n"
                                "data: numpy_array\n"
                                "   data as in a [row, col, N] layout\n"
                                "x: numpy_array\n"
                                "   trimbit values 1D numpy_array[N]\n"
                                "target: numpy_array\n"
                                "   target number of counts as in a numpy_array[row, col] layout\n"
                                "par: numpy_array\n"
                                "   Initial parametes in a numpy_array[npar]\n\n"
                                "Returns\n"
                                "----------\n"
                                "result: numpy_array\n"
                                "   Fit result in numpy array[row, col, npar+1]\n\n");
PyDoc_STRVAR(fit_doc, "fit(data, x, par)\n"
                      "--\n"
                      "\n"
                      "Fit all pixels in the data 3d array.\n\n"
                      "Parameters\n"
                      "----------\n"
                      "data: numpy_array\n"
                      "   data as in a [row, col, N] layout\n"
                      "x: numpy_array[N]\n"
                      "   normally threshold but can be anything that should go on the x axis\n"
                      "par: numpy_array\n"
                      "   Initial parametes in a array[npar]\n\n"
                      "Returns\n"
                      "----------\n"
                      "result: numpy_array\n"
                      "   Fit result in numpy array[row, col, npar]\n\n");

/* Available functions */
static PyObject *fit(PyObject *self, PyObject *args);
static PyObject *find_trimbits(PyObject *self, PyObject *args);
static PyObject *vrf_fit(PyObject *self, PyObject *args);
static PyObject *hist(PyObject *self, PyObject *args);
static PyObject *hist3d(PyObject *self, PyObject *args);
static PyObject *sparse_file_to_th2(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *gaus_float(PyObject *self, PyObject *args);
static PyObject *charge_sharing_func(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, fit_doc},
    {"find_trimbits", find_trimbits, METH_VARARGS, find_trimbits_doc},
    {"vrf_fit", vrf_fit, METH_VARARGS, find_trimbits_doc},
    {"hist", hist, METH_VARARGS, find_trimbits_doc},
    {"hist3d", hist3d, METH_VARARGS, find_trimbits_doc},
    {"sparse_file_to_th2", (PyCFunction)(void(*)(void))sparse_file_to_th2, METH_VARARGS | METH_KEYWORDS, find_trimbits_doc},
    {"gaus_float", gaus_float, METH_VARARGS, find_trimbits_doc},
    {"charge_sharing_func", charge_sharing_func, METH_VARARGS, find_trimbits_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef sls_cmodule_def = {PyModuleDef_HEAD_INIT, "_sls_cmodule",
                                             module_docstring, -1, module_methods};

/* Initialize the module */
PyMODINIT_FUNC PyInit__sls_cmodule(void) {
    PyObject *m = PyModule_Create(&sls_cmodule_def);
    if (m == NULL) {
        return NULL;
    }

    /* Load `numpy` functionality. */
    import_array();
    TPython::Import("ROOT");
    return m;
}

static PyObject *find_trimbits(PyObject *self, PyObject *args) {

    // PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *data_obj;
    PyObject *par_obj;
    PyObject *target_obj;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "OOOO", &data_obj, &x_obj, &target_obj, &par_obj)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *target_array = PyArray_FROM_OTF(target_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    // Exception if it fails
    if (x_array == NULL || data_array == NULL || par_array == NULL || target_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    //! TODO Find this automatically
    int npar = 6;

    // Check that we have the righ number of parameters
    if ((int)PyArray_NDIM((PyArrayObject *)par_array) != 1) {
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }
    if ((int)PyArray_DIM((PyArrayObject *)par_array, 0) != npar) {
        std::cout << "Wrong number of parameters!" << std::endl;
        return NULL;
    }

    // Number of dimensions in the numpy array
    int ndim = (int)PyArray_NDIM((PyArrayObject *)data_array);
    if (ndim != 3) {
        std::cout << "We need a 3D array!";
        return NULL;
    }

    // Get the dimensions of the array
    int shape[3];
    for (int i = 0; i < ndim; i++) {
        shape[i] = (int)PyArray_DIM((PyArrayObject *)data_array, i);
    }

    /* Get a pointer to the data as C-types. */
    double *x = (double *)PyArray_DATA((PyArrayObject *)x_array);
    double *data = (double *)PyArray_DATA((PyArrayObject *)data_array);
    double *target = (double *)PyArray_DATA((PyArrayObject *)target_array);
    double *initpar = (double *)PyArray_DATA((PyArrayObject *)par_array);

    /* Create a numpy array to return to Python */
    npy_intp dims[3] = {shape[0], shape[1], npar + 1};
    PyObject *result_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);

    /* Get a pointer to the data as C-types. */
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);

    // Fit the data
    fit_trimbits(data, x, target, shape, initpar, result);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(data_array);
    Py_DECREF(par_array);

    return result_array;
}

static PyObject *fit(PyObject *self, PyObject *args) {
    // PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *data_obj;
    PyObject *par_obj;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "OOO", &data_obj, &x_obj, &par_obj)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // Exception if it fails
    if (x_array == NULL || data_array == NULL || par_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    //! TODO Find this automatically
    const int npar = 6;

    // Check that we have the righ number of parameters
    if ((int)PyArray_NDIM((PyArrayObject *)par_array) != 1) {
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }
    if ((int)PyArray_DIM((PyArrayObject *)par_array, 0) != npar) {
        std::cout << "Wrong number of parameters!" << std::endl;
        return NULL;
    }

    // Number of dimensions in the numpy array
    int ndim = (int)PyArray_NDIM((PyArrayObject *)data_array);
    if (ndim != 3) {
        std::cout << "We need a 3D array!";
        return NULL;
    }

    // Get the dimensions of the array
    int shape[3];
    for (int i = 0; i < ndim; i++) {
        shape[i] = (int)PyArray_DIM((PyArrayObject *)data_array, i);
    }

    /* Get a pointer to the data as C-types. */
    double *x = (double *)PyArray_DATA((PyArrayObject *)x_array);
    double *data = (double *)PyArray_DATA((PyArrayObject *)data_array);
    double *initpar = (double *)PyArray_DATA((PyArrayObject *)par_array);

    /* Create a numpy array to return to Python */
    npy_intp dims[3] = {shape[0], shape[1], npar};
    PyObject *result_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);

    /* Get a pointer to the data as C-types. */
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);

    // Fit the data
    fit_using_tgraph(data, x, shape, initpar, result);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(data_array);
    Py_DECREF(par_array);

    return result_array;
}

static PyObject *vrf_fit(PyObject *self, PyObject *args) {
    // PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *y_obj;
    PyObject *par_obj;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "OOO", &x_obj, &y_obj, &par_obj)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    // Exception if it fails
    if (x_array == NULL || y_array == NULL || par_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    //! TODO Find this automatically
    const int npar = 3;

    // Check that we have the righ number of parameters
    if ((int)PyArray_NDIM((PyArrayObject *)par_array) != 1) {
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }

    const int n = (int)PyArray_DIM((PyArrayObject *)x_array, 0);
    // std::cout <<"size: "<< n <<std::endl;

    /* Get a pointer to the data as C-types. */
    double *x = (double *)PyArray_DATA((PyArrayObject *)x_array);
    double *y = (double *)PyArray_DATA((PyArrayObject *)y_array);
    double *lim = (double *)PyArray_DATA((PyArrayObject *)par_array);
    double xmin = lim[0];
    double xmax = lim[1];

    /* Create a numpy array to return to Python */
    npy_intp dims[1] = {npar};
    PyObject *result_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    /* Get a pointer to the data as C-types. */
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);

    // Fit the data
    // fit_using_tgraph(data, x, shape, initpar, result);
    gaus_fit(n, x, y, xmin, xmax, result);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(par_array);

    return result_array;
}

static PyObject *gaus_float(PyObject *self, PyObject *args) {
    // PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *y_obj;
    PyObject *par_obj;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "OOO", &x_obj, &y_obj, &par_obj)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    // PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    // PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    // Exception if it fails
    if (x_array == NULL || y_array == NULL || par_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    //! TODO Find this automatically
    const int npar = 3;

    // Check that we have the righ number of parameters
    if ((int)PyArray_NDIM((PyArrayObject *)par_array) != 1) {
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }

    const int n = (int)PyArray_DIM((PyArrayObject *)x_array, 0);
    const int n_fits = (int)PyArray_DIM((PyArrayObject *)y_array, 0);

    std::cout << "n_fits: " << n_fits << std::endl;

    /* Get a pointer to the data as C-types. */
    auto *x = (double *)PyArray_DATA((PyArrayObject *)x_array);
    auto *y = (double *)PyArray_DATA((PyArrayObject *)y_array);
    double *lim = (double *)PyArray_DATA((PyArrayObject *)par_array);
    double xmin = lim[0];
    double xmax = lim[1];

    /* Create a numpy array to return to Python */
    npy_intp dims[2] = {n_fits, npar};
    PyObject *result_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    /* Get a pointer to the data as C-types. */
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_array);

    // Fit the data
    // fit_using_tgraph(data, x, shape, initpar, result);
    for (int i = 0; i != n_fits; ++i)
        gaus_fit2(n, x, y + (i * n), xmin, xmax, result + (i * npar));

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(par_array);

    return result_array;
}

static PyObject *hist(PyObject *self, PyObject *args) {
    // PyObject to be extracted from *args holds data and parameters to the TH1D
    PyObject *data_obj;
    PyObject *par_obj;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "OO", &data_obj, &par_obj)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    // Exception if it fails
    if (data_array == NULL || par_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    // is the length of par 3? [xmin, xmax, bins]
    const int npar = 3;
    if ((int)PyArray_SIZE((PyArrayObject *)par_array) != npar) {
        std::cout << "Wrong length of parameters" << std::endl;
        return NULL;
    }

    /* Get a pointer to the data as C-types. */
    double *data = (double *)PyArray_DATA((PyArrayObject *)data_array);
    double *lim = (double *)PyArray_DATA((PyArrayObject *)par_array);
    double xmin = lim[0];
    double xmax = lim[1];
    int bins = static_cast<int>(lim[2]);

    // Histogram goes here
    auto h = new TH1D("test", "test", bins, xmin, xmax);

    // Number of elements in the data array
    const int n_elements = (int)PyArray_SIZE((PyArrayObject *)data_array);

    // Fill the histogram
    for (int i = 0; i < n_elements; ++i)
        h->Fill(data[i]);

    // Create numpy array for result
    npy_intp dims[1] = {bins};
    PyObject *x_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *y_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    /* Get a pointer to the data as C-types. */
    double *x = (double *)PyArray_DATA((PyArrayObject *)x_array);
    double *y = (double *)PyArray_DATA((PyArrayObject *)y_array);

    //    //Loop and fetch data
    for (int i = 0; i < bins; ++i) {
        // bin 0 is underflow starting with bin 1
        x[i] = h->GetBinLowEdge(i + 1);
        y[i] = h->GetBinContent(i + 1);
    }

    // Dict to use with returns dict['x'] etc.
    auto dict = PyDict_New();
    PyDict_SetItemString(dict, "x", x_array);
    PyDict_SetItemString(dict, "y", y_array);
    PyDict_SetItemString(dict, "mean", Py_BuildValue("f", h->GetMean()));
    PyDict_SetItemString(dict, "std", Py_BuildValue("f", h->GetStdDev()));

    // Clean up
    delete h;
    Py_DECREF(data_array);
    Py_DECREF(par_array);

    return dict;
}

char* uuid(char out[UUID_STR_LEN]){
  uuid_t b;
  uuid_generate(b);
  uuid_unparse_lower(b, out);
  return out;
}

// for the moment constrained
struct __attribute__((__packed__)) Hit
{
    int16_t row;
    int16_t col;
    double energy;
};


static PyObject *sparse_file_to_th2(PyObject *self, PyObject *args, PyObject *kwds) {


    int nbins = 0;
    double xmin = 0;
    double xmax = 0;

    // Parse file name, accepts string or pathlike objects
    PyObject *fname_obj = NULL;
    PyObject *fname_bytes = NULL;
    char *fname = NULL;
    Py_ssize_t len;

    static char *kwlist[] = {"fname", "nbins", "xmin", "xmax", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oidd", kwlist, &fname_obj,
                                     &nbins, &xmin, &xmax)) {
        return NULL;
    }

    if (fname_obj != Py_None)
        if (!PyUnicode_FSConverter(fname_obj, &fname_bytes))
            return NULL;

    PyBytes_AsStringAndSize(fname_bytes, &fname, &len);


    char histogram_name[UUID_STR_LEN]={0};
    uuid(histogram_name);
    std::cout << "histogram: " << histogram_name << "\n";

    FILE* fp  = fopen(fname, "rb");

    if (fp == NULL) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, fname);
        return NULL;
    }

    int32_t magic, nrow, ncol, itemsize;
    if(!fread(&magic, sizeof(magic), 1, fp))
        return NULL;
    if(!fread(&nrow, sizeof(nrow), 1, fp))
        return NULL;
    if(!fread(&ncol, sizeof(ncol), 1, fp))
        return NULL;
    if(!fread(&itemsize, sizeof(itemsize), 1, fp))
        return NULL;


    // Histogram goes here
    int pixel_bins = nrow*ncol;
    double pixel_min = -.5;
    double pixel_xmax = pixel_bins+.5;
    auto h = new TH2D(histogram_name, "test", pixel_bins, pixel_min , pixel_xmax, nbins, xmin, xmax);

    // Fill the histogram
    Hit hit{};
    while(fread(&hit, sizeof(hit), 1, fp)){
        size_t pos = hit.col+(hit.row*ncol);
        // size_t pos = hit.row+(hit.col*nrow);

        if(hit.energy>xmin){
            // std::cout << "row: " << hit.row << " col: " << hit.col << " pos: " << pos << " energy: " << hit.energy << std::endl;
            h->Fill(pos, hit.energy);
            // counter++;
        }   
    }

    return TPython::CPPInstance_FromVoidPtr(h, h->ClassName(), kTRUE);
}




static PyObject *hist3d(PyObject *self, PyObject *args) {
    // PyObject to be extracted from *args holds data and parameters to the TH1D
    PyObject *data_obj = NULL;

    int nbins = 0;
    double xmin = 0;
    double xmax = 0;
    const char* name = nullptr;

    // Check and parse..
    if (!PyArg_ParseTuple(args, "Osidd", &data_obj, &name, &nbins, &xmin, &xmax)) {
        return NULL;
    }

    // Numpy array from the parsed objects
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);


    // Exception if it fails
    if (data_array == NULL) {
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl;
        return NULL;
    }

    // Number of dimensions in the numpy array
    int ndim = (int)PyArray_NDIM((PyArrayObject *)data_array);
    if (ndim != 3) {
        std::cout << "We need a 3D array!";
        return NULL;
    }

    // Get the dimensions of the array
    int shape[3];
    for (int i = 0; i < ndim; i++) {
        shape[i] = (int)PyArray_DIM((PyArrayObject *)data_array, i);
    }

    double *data = (double *)PyArray_DATA((PyArrayObject *)data_array);

    // Histogram goes here
    int pixel_bins = shape[1]*shape[2];
    double pixel_min = -.5;
    double pixel_xmax = shape[1]*shape[2];
    auto h = new TH2D(name, "test", pixel_bins, pixel_min , pixel_xmax, nbins, xmin, xmax);

    // Time to do some filling

    for(size_t frame=0; frame!=shape[0]; ++frame){
        for (size_t row=0; row!=shape[1]; ++row){
            for (size_t col=0; col!=shape[2]; ++col){
                size_t pos = col+(row*shape[2]);
                if(*data>xmin)
                    h->Fill(pos, *data);
                ++data;
            }
        }
    }

    Py_DECREF(data_array);

    // Create python object to return
    return TPython::CPPInstance_FromVoidPtr(h, h->ClassName(), kTRUE);

}

static PyObject *charge_sharing_func(PyObject *self, PyObject *args){
    double xmin = 2000;
    double xmax = 3500;
    if (!PyArg_ParseTuple(args, "|dd", &xmin, &xmax)) {
        return NULL;
    }
    auto func = new TF1("charge", charge_sharing_model, xmin,xmax,4);
    return TPython::CPPInstance_FromVoidPtr(func, func->ClassName(), kTRUE);
}