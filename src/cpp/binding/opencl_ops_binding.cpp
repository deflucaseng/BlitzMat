#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "operation_manager.hpp"
#include <numpy/arrayobject.h>

typedef struct
{
    PyObject_HEAD OperationManager *op_manager;
} PyOperationManager;

static void
PyOperationManager_dealloc(PyOperationManager *self)
{
    if (self->op_manager)
    {
        delete self->op_manager;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
PyOperationManager_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyOperationManager *self;
    self = (PyOperationManager *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->op_manager = NULL;
    }
    return (PyObject *)self;
}

static int
PyOperationManager_init(PyOperationManager *self, PyObject *args, PyObject *kwds)
{
    const char *device_type_str;
    if (!PyArg_ParseTuple(args, "s", &device_type_str))
    {
        return -1;
    }

    OperationManager::device_types device_type;
    if (strcmp(device_type_str, "CPU") == 0)
    {
        device_type = OperationManager::device_types::CPU_DEVICE;
    }
    else if (strcmp(device_type_str, "GPU") == 0)
    {
        device_type = OperationManager::device_types::GPU_DEVICE;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Device type must be either 'CPU' or 'GPU'");
        return -1;
    }

    try
    {
        self->op_manager = new OperationManager(device_type);
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    return 0;
}

static PyObject *
PyOperationManager_multi_vector_op(PyOperationManager *self, PyObject *args)
{
    PyArrayObject *lhs_array, *rhs_array;
    const char *op_type_str;

    if (!PyArg_ParseTuple(args, "sOO", &op_type_str, &lhs_array, &rhs_array))
    {
        return NULL;
    }

    if (!PyArray_Check(lhs_array) || !PyArray_Check(rhs_array))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }

    // Verify array types
    if (PyArray_TYPE(lhs_array) != NPY_float || PyArray_TYPE(rhs_array) != NPY_float)
    {
        PyErr_SetString(PyExc_TypeError, "Arrays must be of type numpy.float64");
        return NULL;
    }

    // Get array dimensions
    int lheight = PyArray_DIM(lhs_array, 0);
    int lwidth = PyArray_DIM(lhs_array, 1);
    int rheight = PyArray_DIM(rhs_array, 0);
    int rwidth = PyArray_DIM(rhs_array, 1);

    // Get operation type
    operation_types op_type;
    if (strcmp(op_type_str, "add") == 0)
    {
        op_type = operation_types::ELEM_WISE_ADD;
    }
    else if (strcmp(op_type_str, "subtract") == 0)
    {
        op_type = operation_types::ELEM_WISE_SUB;
    }
    else if (strcmp(op_type_str, "divide") == 0)
    {
        op_type = operation_types::ELEM_WISE_DIV;
    }
    else if (strcmp(op_type_str, "multiply") == 0)
    {
        op_type = operation_types::ELEM_WISE_MUL;
    }
    else if (strcmp(op_type_str, "matrix_multiply") == 0)
    {
        op_type = operation_types::MATRIX_MULTIPLICATION;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid operation type");
        return NULL;
    }

    // Get data pointers
    float *lhs_data = (float *)PyArray_DATA(lhs_array);
    float *rhs_data = (float *)PyArray_DATA(rhs_array);

    try
    {
        float *result = self->op_manager->multi_vector_op(
            op_type, lhs_data, lheight, lwidth, rhs_data, rheight, rwidth);

        // Create output numpy array
        npy_intp dims[2] = {lheight, rwidth};
        PyObject *result_array = PyArray_SimpleNewFromData(2, dims, NPY_float, result);

        // Set array to own the memory
        PyArray_ENABLEFLAGS((PyArrayObject *)result_array, NPY_ARRAY_OWNDATA);

        return result_array;
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *
PyOperationManager_single_vector_op(PyOperationManager *self, PyObject *args)
{
    PyArrayObject *data_array;
    const char *op_type_str;

    if (!PyArg_ParseTuple(args, "sO", &op_type_str, &data_array))
    {
        return NULL;
    }

    if (!PyArray_Check(data_array))
    {
        PyErr_SetString(PyExc_TypeError, "Argument must be a numpy array");
        return NULL;
    }

    if (PyArray_TYPE(data_array) != NPY_float)
    {
        PyErr_SetString(PyExc_TypeError, "Array must be of type numpy.float64");
        return NULL;
    }

    int height = PyArray_DIM(data_array, 0);
    int width = PyArray_DIM(data_array, 1);
    float *data = (float *)PyArray_DATA(data_array);

    operation_types op_type;
    if (strcmp(op_type_str, "transpose") == 0)
    {
        op_type = operation_types::TRANSPOSE;
    }
    else if (strcmp(op_type_str, "inverse") == 0)
    {
        op_type = operation_types::INVERSE;
    }
    else if (strcmp(op_type_str, "trace") == 0)
    {
        op_type = operation_types::TRACE;
    }
    else if (strcmp(op_type_str, "frobenius_norm") == 0)
    {
        op_type = operation_types::FROBENIUS_NORM;
    }
    else if (strcmp(op_type_str, "determinant") == 0)
    {
        op_type = operation_types::DETERMINANT;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid operation type");
        return NULL;
    }

    try
    {
        float *result = self->op_manager->single_vector_op(op_type, data, height, width);

        npy_intp dims[2] = {width, height}; // Note: dimensions swapped for transpose
        PyObject *result_array = PyArray_SimpleNewFromData(2, dims, NPY_float, result);
        PyArray_ENABLEFLAGS((PyArrayObject *)result_array, NPY_ARRAY_OWNDATA);

        return result_array;
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyMethodDef PyOperationManager_methods[] = {
    {"multi_vector_op", (PyCFunction)PyOperationManager_multi_vector_op, METH_VARARGS,
     "Perform operation on two vectors"},
    {"single_vector_op", (PyCFunction)PyOperationManager_single_vector_op, METH_VARARGS,
     "Perform operation on a single vector"},
    {NULL} /* Sentinel */
};

static PyTypeObject PyOperationManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0) "opencl_ops.OperationManager", /* tp_name */
    sizeof(PyOperationManager),                                   /* tp_basicsize */
    0,                                                            /* tp_itemsize */
    (destructor)PyOperationManager_dealloc,                       /* tp_dealloc */
    0,                                                            /* tp_print */
    0,                                                            /* tp_getattr */
    0,                                                            /* tp_setattr */
    0,                                                            /* tp_reserved */
    0,                                                            /* tp_repr */
    0,                                                            /* tp_as_number */
    0,                                                            /* tp_as_sequence */
    0,                                                            /* tp_as_mapping */
    0,                                                            /* tp_hash  */
    0,                                                            /* tp_call */
    0,                                                            /* tp_str */
    0,                                                            /* tp_getattro */
    0,                                                            /* tp_setattro */
    0,                                                            /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                     /* tp_flags */
    "OpenCL Operation Manager",                                   /* tp_doc */
    0,                                                            /* tp_traverse */
    0,                                                            /* tp_clear */
    0,                                                            /* tp_richcompare */
    0,                                                            /* tp_weaklistoffset */
    0,                                                            /* tp_iter */
    0,                                                            /* tp_iternext */
    PyOperationManager_methods,                                   /* tp_methods */
    0,                                                            /* tp_members */
    0,                                                            /* tp_getset */
    0,                                                            /* tp_base */
    0,                                                            /* tp_dict */
    0,                                                            /* tp_descr_get */
    0,                                                            /* tp_descr_set */
    0,                                                            /* tp_dictoffset */
    (initproc)PyOperationManager_init,                            /* tp_init */
    0,                                                            /* tp_alloc */
    PyOperationManager_new,                                       /* tp_new */
};

static PyModuleDef opencl_ops_module = {
    PyModuleDef_HEAD_INIT,
    "opencl_ops",
    "OpenCL Operations Module",
    -1,
    NULL};

PyMODINIT_FUNC
PyInit_opencl_ops(void)
{
    import_array(); // Initialize NumPy

    PyObject *m;
    if (PyType_Ready(&PyOperationManagerType) < 0)
        return NULL;

    m = PyModule_Create(&opencl_ops_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyOperationManagerType);
    if (PyModule_AddObject(m, "OperationManager", (PyObject *)&PyOperationManagerType) < 0)
    {
        Py_DECREF(&PyOperationManagerType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}