#define  NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include "fit_tgraph.h"


/* Docstrings */
static char module_docstring[] =
    "sls_cmodule provies access to compiled c++ and ROOT functions supporting "\
    "detector calibration. The library uses numpy arrays as input and return values.";

//TODO! update!
static char load_docstring[] =
    "Loading binary values from a file";

static char fit_docstring[] =
    "Fit scurve to all pixels";

PyDoc_STRVAR(
    find_trimbits_doc,
    "find_trimbits(data, x, target, par)\n"
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
    "   Fit result in numpy array[row, col, npar+1]\n\n"
        );
PyDoc_STRVAR(
    fit_doc,
    "fit(data, x, par)\n"
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
    "   Fit result in numpy array[row, col, npar]\n\n"
        );


/* Available functions */
static PyObject *fit(PyObject *self, PyObject *args);
static PyObject *find_trimbits(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, fit_doc},
    {"find_trimbits", find_trimbits, METH_VARARGS, find_trimbits_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sls_cmodule_def = {
    PyModuleDef_HEAD_INIT,
    "_sls_cmodule",
    module_docstring,
    -1,
    module_methods
};

/* Initialize the module */
PyMODINIT_FUNC
PyInit__sls_cmodule(void)
{
    PyObject *m = PyModule_Create(&sls_cmodule_def);
    if (m == NULL){
        return NULL;
        }

    /* Load `numpy` functionality. */
    import_array();
    return m;
}

static PyObject *find_trimbits(PyObject *self, PyObject *args){
    
    //PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *data_obj;
    PyObject *par_obj;
    PyObject *target_obj;
    
    //Check and parse..
    if (!PyArg_ParseTuple(args, "OOOO", &data_obj, &x_obj, &target_obj, &par_obj)){
        return NULL;
    }    

    //Numpy array from the parsed objects 
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *target_array = PyArray_FROM_OTF(target_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    //Exception if it fails
    if (x_array == NULL || data_array == NULL || par_array == NULL || target_array == NULL ){
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl; 
        return NULL;
    }
    
    //!TODO Find this automatically 
    int npar = 6;

    //Check that we have the righ number of parameters
    if ( (int)PyArray_NDIM( (PyArrayObject*)par_array) != 1 ){
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }
    if ( (int)PyArray_DIM((PyArrayObject*)par_array, 0) != npar ){
        std::cout << "Wrong number of parameters!" << std::endl;
        return NULL;
    }


    //Number of dimensions in the numpy array
    int ndim = (int)PyArray_NDIM( (PyArrayObject*)data_array);
    if (ndim != 3){
        std::cout << "We need a 3D array!";
        return NULL;
    }

    //Get the dimensions of the array
    int shape[3];
    for (int i = 0; i< ndim; i++){
        shape[i] = (int)PyArray_DIM((PyArrayObject*)data_array, i);
    }


    /* Get a pointer to the data as C-types. */
    double *x = (double*)PyArray_DATA((PyArrayObject*)x_array);
    double *data = (double*)PyArray_DATA((PyArrayObject*)data_array);
    double *target = (double*)PyArray_DATA((PyArrayObject*)target_array);
    double *initpar = (double*)PyArray_DATA((PyArrayObject*)par_array);
    
    
    /* Create a numpy array to return to Python */
    npy_intp dims[3] = { shape[0], shape[1], npar +1 };
    PyObject *result_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    

    /* Get a pointer to the data as C-types. */
    double *result = (double*)PyArray_DATA((PyArrayObject*)result_array);

    //Fit the data
    fit_trimbits(data, x, target, shape, initpar, result);


    //Clean up
    Py_DECREF(x_array);
    Py_DECREF(data_array);
    Py_DECREF(par_array);

    return result_array;
}

static PyObject *fit(PyObject *self, PyObject *args)
{
    //PyObject to be extracted from *args
    PyObject *x_obj;
    PyObject *data_obj;
    PyObject *par_obj;

    //Check and parse..
    if (!PyArg_ParseTuple(args, "OOO", &data_obj, &x_obj, &par_obj)){
        return NULL;
    }

    //Numpy array from the parsed objects 
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *data_array = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
    PyObject *par_array = PyArray_FROM_OTF(par_obj, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);

    //Exception if it fails
    if (x_array == NULL || data_array == NULL || par_array == NULL ){
        std::cout << "Something went wrong, possibly crappy arguments?" << std::endl; 
        return NULL;
    }



    //!TODO Find this automatically 
    int npar = 6;

    //Check that we have the righ number of parameters
    if ( (int)PyArray_NDIM( (PyArrayObject*)par_array) != 1 ){
        std::cout << "ndimpar!" << std::endl;
        return NULL;
    }
    if ( (int)PyArray_DIM((PyArrayObject*)par_array, 0) != npar ){
        std::cout << "Wrong number of parameters!" << std::endl;
        return NULL;
    }


    //Number of dimensions in the numpy array
    int ndim = (int)PyArray_NDIM( (PyArrayObject*)data_array);
    if (ndim != 3){
        std::cout << "We need a 3D array!";
        return NULL;
    }

    //Get the dimensions of the array
    int shape[3];
    for (int i = 0; i< ndim; i++){
        shape[i] = (int)PyArray_DIM((PyArrayObject*)data_array, i);
    }


    /* Get a pointer to the data as C-types. */
    double *x = (double*)PyArray_DATA((PyArrayObject*)x_array);
    double *data = (double*)PyArray_DATA((PyArrayObject*)data_array);
    double *initpar = (double*)PyArray_DATA((PyArrayObject*)par_array);

    /* Create a numpy array to return to Python */
    npy_intp dims[3] = { shape[0], shape[1], npar };
    PyObject *result_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);


    /* Get a pointer to the data as C-types. */
    double *result = (double*)PyArray_DATA((PyArrayObject*)result_array);

    //Fit the data
    fit_using_tgraph(data, x, shape, initpar, result);


    //Clean up
    Py_DECREF(x_array);
    Py_DECREF(data_array);
    Py_DECREF(par_array);

    return result_array;
}
