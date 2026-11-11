// Shared Python boundary for all native modules; numerical kernels use no Python layouts.
#pragma once
#if !defined(Py_LIMITED_API) || Py_LIMITED_API != 0x030B0000
#error "algroots native modules require the CPython 3.11 Limited API"
#endif
#include <Python.h>

// PySequence_Fast returns a list or tuple. These Stable-ABI accessors retain
// borrowed references, avoiding an allocation or INCREF/DECREF per input term.
static inline Py_ssize_t native_sequence_size(PyObject *sequence) {
  return PyList_Check(sequence) ? PyList_Size(sequence) : PyTuple_Size(sequence);
}
static inline PyObject *native_sequence_item(PyObject *sequence, Py_ssize_t index) {
  return PyList_Check(sequence) ? PyList_GetItem(sequence, index)
                                : PyTuple_GetItem(sequence, index);
}
