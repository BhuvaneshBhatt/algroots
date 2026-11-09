// CPython bindings, parsing and result ownership. Included in one translation
// unit.
static Mon read_m(PyObject *o, unsigned n) {
  if (!PyTuple_Check(o))
    throw std::runtime_error("monomial must be tuple");
  Py_ssize_t k = PyTuple_GET_SIZE(o);
  if (k != (Py_ssize_t)n)
    throw std::runtime_error("invalid monomial width");
  std::vector<unsigned> m;
  m.reserve(k);
  for (Py_ssize_t i = 0; i < k; ++i) {
    auto v = PyLong_AsLongLong(PyTuple_GET_ITEM(o, i));
    if (PyErr_Occurred() || v < 0 || v > UINT32_MAX) {
      preserve_python_interrupt();
      PyErr_Clear();
      throw std::runtime_error("bad exponent");
    }
    m.push_back((unsigned)v);
  }
  return Mon(std::move(m));
}
static Poly read_poly(PyObject *o, const Engine &e) {
  if (!PyDict_Check(o))
    throw std::runtime_error("expected sparse polynomial dictionary");
  Poly p;
  Py_ssize_t pos = 0;
  PyObject *k, *v;
  while (PyDict_Next(o, &pos, &k, &v)) {
    Mon m = read_m(k, e.n);
    long long c = PyLong_AsLongLong(v);
    if (PyErr_Occurred()) {
      preserve_python_interrupt();
      PyErr_Clear();
      throw std::runtime_error("invalid modular coefficient");
    }
    uint64_t x = (c % (int64_t)e.p + (int64_t)e.p) % e.p;
    if (x)
      p[std::move(m)] = x;
  }
  return p;
}
static PyObject *make_m(const Mon &m) {
  auto t = own_python(PyTuple_New(m.size()));
  for (size_t i = 0; i < m.size(); ++i) {
    auto value = own_python(PyLong_FromUnsignedLong(m[i]));
    PyTuple_SET_ITEM(t.get(), i, value.release());
  }
  return t.release();
}
static PyObject *make_poly(const Poly &p) {
  auto d = own_python(PyDict_New());
  for (const auto &[m, c] : p) {
    auto k = own_python(make_m(m));
    auto v = own_python(PyLong_FromUnsignedLongLong(c));
    check_python(PyDict_SetItem(d.get(), k.get(), v.get()));
  }
  return d.release();
}

static PyObject *compute(PyObject *module_object, PyObject *args,
                         PyObject *kw) {
  PyObject *polynomials;
  int p, n, collect_batch_metrics = 0;
  const char *ord = "grevlex";
  Py_ssize_t maxpairs = 20000, maxterms = 100000, maxbasis = 256, batch = 128,
             max_reduction_steps = 250000;
  static const char *keys[] = {"polynomials",
                               "prime",
                               "nvars",
                               "order",
                               "max_pairs",
                               "max_terms",
                               "max_basis",
                               "batch_size",
                               "max_reduction_steps",
                               "collect_batch_metrics",
                               nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kw, "Oii|snnnnnp", (char **)keys, &polynomials, &p, &n, &ord,
          &maxpairs, &maxterms, &maxbasis, &batch, &max_reduction_steps,
          &collect_batch_metrics))
    return nullptr;
  try {
    if (p < 2 || p > 1000000007 || n < 1 || n > 64 || maxpairs < 1 ||
        maxterms < 1 || maxbasis < 1 || batch < 1 || max_reduction_steps < 1)
      throw std::runtime_error("invalid native F4 limits");
    // The public adapter checks primality. Also validate direct extension
    // calls.
    for (int divisor = 2; static_cast<int64_t>(divisor) * divisor <= p;
         ++divisor)
      if (p % divisor == 0)
        throw std::runtime_error("modulus must be prime");
    int order = std::string(ord) == "lex"       ? 0
                : std::string(ord) == "grlex"   ? 1
                : std::string(ord) == "grevlex" ? 2
                                                : -1;
    if (order < 0)
      throw std::runtime_error("unsupported term order");
    struct PolyOrderScope {
      int before;
      explicit PolyOrderScope(int value) : before(active_poly_order) {
        active_poly_order = value;
      }
      ~PolyOrderScope() { active_poly_order = before; }
    } order_scope(order);
    MonPool pool;
    MonPoolScope monomial_scope(pool);
    Engine e{static_cast<unsigned>(n),
             static_cast<unsigned>(p),
             order,
             static_cast<size_t>(maxpairs),
             static_cast<size_t>(maxterms),
             static_cast<size_t>(maxbasis),
             std::max(size_t{128}, static_cast<size_t>(batch))};
    e.max_nf_steps = static_cast<size_t>(max_reduction_steps);
    e.collect_batch_metrics = collect_batch_metrics != 0;
    auto inputs = own_python(
        PySequence_Fast(polynomials, "polynomials must be a sequence"));
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(inputs.get()); ++i)
      insert(e, read_poly(PySequence_Fast_GET_ITEM(inputs.get(), i), e));
    inputs.reset();
    {
      NativeExecution execution;
      run(e);
    }
    auto basis = own_python(PyList_New(0)),
         statistics = own_python(PyDict_New());
    for (const auto &f : e.basis) {
      auto item = own_python(make_poly(f));
      check_python(PyList_Append(basis.get(), item.get()));
    }
    auto stats = statistics.get();
    auto setstat = [&](const char *name, size_t value) {
      auto item = own_python(PyLong_FromSize_t(value));
      check_python(PyDict_SetItemString(stats, name, item.get()));
    };
#include "_native_f4_stats.inc"
    return own_python(PyTuple_Pack(2, basis.get(), stats)).release();
  } catch (const NativePythonError &) {
    return nullptr;
  } catch (const NativeBudgetError &ex) {
    PyObject *error = PyObject_GetAttrString(module_object, "LimitError");
    if (!error)
      return nullptr;
    PyErr_SetString(error, ex.what());
    Py_DECREF(error);
    return nullptr;
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  } catch (const std::exception &ex) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_RuntimeError, ex.what());
    return nullptr;
  }
}
static PyMethodDef methods[] = {
    {"compute", (PyCFunction)compute, METH_VARARGS | METH_KEYWORDS,
     "Reduced GF(p) Groebner basis using tiled F4 and exact final reduction."},
    {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_f4", nullptr, -1,
                             methods};
PyMODINIT_FUNC PyInit__native_f4(void) {
  PyObject *result = PyModule_Create(&module);
  if (!result)
    return nullptr;
  PyObject *error = PyErr_NewException("algroots._native_f4.LimitError",
                                       PyExc_RuntimeError, nullptr);
  if (!error || PyModule_AddObjectRef(result, "LimitError", error) < 0) {
    Py_XDECREF(error);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(error);
  return result;
}
