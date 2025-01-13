#ifndef OPENCL_OPS_BINDING_HPP
#define OPENCL_OPS_BINDING_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "operation_manager.hpp"

// Structure for the Python OperationManager object
typedef struct {
    PyObject_HEAD
    OperationManager* op_manager;
} PyOperationManager;

// Deallocation function
static void PyOperationManager_dealloc(PyOperationManager* self);

// Creation and initialization functions
static PyObject* PyOperationManager_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
static int PyOperationManager_init(PyOperationManager* self, PyObject* args, PyObject* kwds);

// Method functions
static PyObject* PyOperationManager_multi_vector_op(PyOperationManager* self, PyObject* args);
static PyObject* PyOperationManager_single_vector_op(PyOperationManager* self, PyObject* args);

// Method definitions array
static PyMethodDef PyOperationManager_methods[];

// Type object
static PyTypeObject PyOperationManagerType;

// Module definition
static PyModuleDef opencl_ops_module;

// Module initialization function
PyMODINIT_FUNC PyInit_opencl_ops(void);

#endif // OPENCL_OPS_BINDING_HPP