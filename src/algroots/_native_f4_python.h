// CPython ownership and cancellation support, outside the arithmetic kernels.
#pragma once
#include <Python.h>
#include <memory>

struct NativePythonError {};
struct PyDecref {
 void operator()(PyObject* value)const noexcept {Py_XDECREF(value);}
};
using PyOwner=std::unique_ptr<PyObject,PyDecref>;
static PyOwner own_python(PyObject* value){
 if(!value)throw NativePythonError{};
 return PyOwner(value);
}
static void check_python(int status){if(status<0)throw NativePythonError{};}

class NativeExecution;
static thread_local NativeExecution* active_native_execution=nullptr;
// A computation uses Python objects only while parsing inputs and exporting
// results. Numerical execution releases the GIL; coarse checkpoints reacquire
// it briefly to deliver signals. All monomial/allocator state is thread-local.
class NativeExecution {
 NativeExecution* previous_;
 PyThreadState* state_;
public:
 NativeExecution():previous_(active_native_execution),state_(PyEval_SaveThread()){
  active_native_execution=this;
 }
 NativeExecution(const NativeExecution&)=delete;
 NativeExecution& operator=(const NativeExecution&)=delete;
 ~NativeExecution(){
  if(state_)PyEval_RestoreThread(state_);
  active_native_execution=previous_;
 }
 void check(){
  // A Python signal handler may reenter compute while this checkpoint has
  // the GIL. Input parsing in that nested call can itself reach a checkpoint.
  if(!state_){if(PyErr_CheckSignals()<0)throw NativePythonError{};return;}
  PyEval_RestoreThread(state_);state_=nullptr;
  if(PyErr_CheckSignals()<0)throw NativePythonError{};
  state_=PyEval_SaveThread();
 }
};
static inline void native_check_signals(){
 if(active_native_execution)active_native_execution->check();
 else if(PyErr_CheckSignals()<0)throw NativePythonError{};
}
static void preserve_python_interrupt(){
 if(PyErr_ExceptionMatches(PyExc_KeyboardInterrupt)||PyErr_ExceptionMatches(PyExc_MemoryError))
  throw NativePythonError{};
}
