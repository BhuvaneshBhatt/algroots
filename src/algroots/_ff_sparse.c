#define PY_SSIZE_T_CLEAN
#include "_python_abi.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* All field products use uint64_t to avoid overflow for p <= 2**30. */
typedef struct { Py_ssize_t col; uint32_t value; } Term;
typedef struct { Term *terms; Py_ssize_t nnz; } Pivot;
typedef struct { Py_ssize_t col; uint32_t value; unsigned char used; } HashEntry;
typedef struct {
    HashEntry *table;
    size_t cap, occupied;
    Py_ssize_t *heap;
    size_t heap_len, heap_cap;
} SparseRow;

static uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t p) {
    return (uint32_t)(((uint64_t)a * b) % p);
}
static uint32_t pow_mod(uint32_t a, uint32_t n, uint32_t p) {
    uint32_t result = 1;
    while (n) {
        if (n & 1U) result = mul_mod(result, a, p);
        a = mul_mod(a, a, p);
        n >>= 1;
    }
    return result;
}
static int append_op(PyObject *ops, Py_ssize_t pivot, uint32_t factor) {
    PyObject *item = Py_BuildValue("(nI)", pivot, factor);
    if (!item) return -1;
    int rc = PyList_Append(ops, item);
    Py_DECREF(item);
    return rc;
}
static uint64_t hash_column(Py_ssize_t column) {
    uint64_t x = (uint64_t)column;
    x ^= x >> 33; x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33; x *= UINT64_C(0xc4ceb9fe1a85ec53);
    return x ^ (x >> 33);
}
static int reserve_hash(SparseRow *r, size_t count) {
    if (r->cap && count * 10 < r->cap * 7) return 0;
    size_t cap = r->cap ? r->cap * 2 : 32;
    while (count * 10 >= cap * 7) cap *= 2;
    if (cap > SIZE_MAX / sizeof(HashEntry)) return -1;
    HashEntry *table = (HashEntry *)PyMem_Calloc(cap, sizeof(HashEntry));
    if (!table) return -1;
    for (size_t i = 0; i < r->cap; ++i) {
        HashEntry old = r->table[i];
        if (!old.used) continue;
        size_t slot = (size_t)hash_column(old.col) & (cap - 1);
        while (table[slot].used) slot = (slot + 1) & (cap - 1);
        table[slot] = old;
    }
    PyMem_Free(r->table); r->table = table; r->cap = cap;
    return 0;
}
static HashEntry *lookup_hash(SparseRow *r, Py_ssize_t col) {
    if (!r->cap) return NULL;
    size_t slot = (size_t)hash_column(col) & (r->cap - 1);
    while (r->table[slot].used) {
        if (r->table[slot].col == col) return r->table + slot;
        slot = (slot + 1) & (r->cap - 1);
    }
    return NULL;
}
static int heap_push(SparseRow *r, Py_ssize_t col) {
    if (r->heap_len == r->heap_cap) {
        size_t size = r->heap_cap ? 2 * r->heap_cap : 32;
        if (size > SIZE_MAX / sizeof(Py_ssize_t)) return -1;
        Py_ssize_t *newheap = PyMem_Realloc(r->heap, size * sizeof(Py_ssize_t));
        if (!newheap) return -1;
        r->heap = newheap; r->heap_cap = size;
    }
    size_t position = r->heap_len++;
    while (position) {
        size_t parent = (position - 1) / 2;
        if (r->heap[parent] <= col) break;
        r->heap[position] = r->heap[parent]; position = parent;
    }
    r->heap[position] = col;
    return 0;
}
static Py_ssize_t heap_pop(SparseRow *r) {
    Py_ssize_t result = r->heap[0], last = r->heap[--r->heap_len];
    if (!r->heap_len) return result;
    size_t i = 0;
    while (2 * i + 1 < r->heap_len) {
        size_t c = 2 * i + 1;
        if (c + 1 < r->heap_len && r->heap[c+1] < r->heap[c]) ++c;
        if (last <= r->heap[c]) break;
        r->heap[i] = r->heap[c]; i = c;
    }
    r->heap[i] = last;
    return result;
}
static int sparse_put(SparseRow *r, Py_ssize_t col, uint32_t value) {
    HashEntry *found = lookup_hash(r, col);
    if (found) {
        uint32_t old = found->value; found->value = value;
        return value && !old ? heap_push(r, col) : 0;
    }
    if (!value) return 0;
    if (reserve_hash(r, r->occupied + 1) < 0) return -1;
    size_t slot = (size_t)hash_column(col) & (r->cap - 1);
    while (r->table[slot].used) slot = (slot + 1) & (r->cap - 1);
    r->table[slot] = (HashEntry){col, value, 1};
    r->occupied++;
    return heap_push(r, col);
}
static void sparse_free(SparseRow *r) {
    PyMem_Free(r->table); PyMem_Free(r->heap);
    memset(r, 0, sizeof(*r));
}
static int compare_terms(const void *a, const void *b) {
    const Term *l = a, *r = b;
    return l->col < r->col ? -1 : l->col > r->col;
}

static PyObject *reduce_rows(PyObject *self, PyObject *args) {
    PyObject *rows_input, *rows = NULL, *output = NULL;
    Py_ssize_t ncols;
    unsigned int prime;
    const char *mode = "adaptive";
    Pivot *pivots = NULL;
    Py_ssize_t *column_pivot = NULL;
    uint32_t *scratch = NULL;
    SparseRow sparse = {0};
    if (!PyArg_ParseTuple(args, "OnI|s:reduce_rows", &rows_input, &ncols, &prime, &mode)) return NULL;
    if (ncols < 0 || ncols > 2000000 || prime < 2 || prime > (1U << 30)) {
        PyErr_SetString(PyExc_ValueError, "invalid finite-field matrix dimensions or modulus"); return NULL;
    }
    if (strcmp(mode,"adaptive") && strcmp(mode,"dense") && strcmp(mode,"sparse")) {
        PyErr_SetString(PyExc_ValueError,"active-column mode must be adaptive, dense or sparse"); return NULL;
    }
    rows = PySequence_Fast(rows_input, "rows must be a sequence of sparse dictionaries");
    if (!rows) return NULL;
    Py_ssize_t nrows = native_sequence_size(rows);
    pivots = PyMem_Calloc((size_t)(nrows ? nrows : 1), sizeof(Pivot));
    column_pivot = PyMem_Malloc((size_t)(ncols ? ncols : 1) * sizeof(Py_ssize_t));
    if (!pivots || !column_pivot) { PyErr_NoMemory(); goto fail; }
    for (Py_ssize_t col=0; col<ncols; ++col) column_pivot[col]=-1;
    output = PyList_New(0);
    if (!output) goto fail;
    for (Py_ssize_t i=0; i<nrows; ++i) {
        PyObject *src = native_sequence_item(rows,i);
        if (!PyDict_Check(src)) { PyErr_SetString(PyExc_TypeError,"matrix rows must be dicts"); goto fail; }
        int use_sparse = !strcmp(mode,"sparse") || (!strcmp(mode,"adaptive") &&
            ncols >= 1024 && PyDict_Size(src) < ncols / 12);
        if (use_sparse) { sparse_free(&sparse); }
        else {
            if (!scratch) scratch = PyMem_Calloc((size_t)(ncols ? ncols : 1), sizeof(uint32_t));
            if (!scratch) { PyErr_NoMemory(); goto fail; }
            memset(scratch, 0, (size_t)ncols * sizeof(uint32_t));
        }
        Py_ssize_t pos=0;
        PyObject *key,*value;
        while (PyDict_Next(src,&pos,&key,&value)) {
            Py_ssize_t col=PyLong_AsSsize_t(key);
            if (col == -1 && PyErr_Occurred()) goto fail;
            if (col < 0 || col >= ncols) { PyErr_SetString(PyExc_ValueError,"matrix column out of bounds"); goto fail; }
            long long coeff = PyLong_AsLongLong(value);
            if (coeff == -1 && PyErr_Occurred()) goto fail;
            long long residue = coeff % (long long)prime;
            if (residue < 0) residue += prime;
            if (use_sparse) {
                if (sparse_put(&sparse,col,(uint32_t)residue)<0) {PyErr_NoMemory();goto fail;}
            } else scratch[col]=(uint32_t)residue;
        }
        PyObject *ops = PyList_New(0);
        if (!ops) goto fail;
        if (use_sparse) {
            while (sparse.heap_len) {
                Py_ssize_t col=heap_pop(&sparse);
                HashEntry *entry = lookup_hash(&sparse,col);
                uint32_t factor = entry ? entry->value : 0;
                Py_ssize_t j=column_pivot[col];
                if (!factor || j<0) continue;
                if (append_op(ops,j,factor)<0) { Py_DECREF(ops); goto fail; }
                Pivot *pivot=&pivots[j];
                for (Py_ssize_t k=0;k<pivot->nnz;++k) {
                    Term term=pivot->terms[k];
                    HashEntry *old=lookup_hash(&sparse,term.col);
                    uint32_t v=old ? old->value : 0;
                    uint32_t sub=mul_mod(factor,term.value,prime);
                    uint32_t next=v>=sub ? v-sub : v+prime-sub;
                    if (sparse_put(&sparse,term.col,next)<0) {Py_DECREF(ops);PyErr_NoMemory();goto fail;}
                }
            }
        } else {
            for (Py_ssize_t col=0;col<ncols;++col) {
                uint32_t factor=scratch[col];
                Py_ssize_t j=column_pivot[col];
                if (!factor || j<0) continue;
                if (append_op(ops,j,factor)<0) {Py_DECREF(ops);goto fail;}
                Pivot *pivot=&pivots[j];
                for (Py_ssize_t k=0;k<pivot->nnz;++k) {
                    Term term=pivot->terms[k];
                    uint32_t sub=mul_mod(factor,term.value,prime), old=scratch[term.col];
                    scratch[term.col]=old>=sub ? old-sub : old+prime-sub;
                }
            }
        }
        Py_ssize_t nnz=0;
        if (use_sparse) {
            for (size_t k=0;k<sparse.cap;++k) if (sparse.table[k].used && sparse.table[k].value) ++nnz;
        } else for (Py_ssize_t col=0;col<ncols;++col) if (scratch[col]) ++nnz;
        Term *terms=NULL;
        if (nnz) {
            terms=PyMem_Malloc((size_t)nnz*sizeof(Term));
            if (!terms) {Py_DECREF(ops);PyErr_NoMemory();goto fail;}
        }
        Py_ssize_t cursor=0;
        if (use_sparse) {
            for (size_t k=0;k<sparse.cap;++k) if (sparse.table[k].used && sparse.table[k].value)
                terms[cursor++]=(Term){sparse.table[k].col,sparse.table[k].value};
            qsort(terms,(size_t)nnz,sizeof(Term),compare_terms);
        } else for (Py_ssize_t col=0;col<ncols;++col) if (scratch[col])
            terms[cursor++]=(Term){col,scratch[col]};
        uint32_t inverse=1;
        if (nnz) {
            inverse=pow_mod(terms[0].value,prime-2,prime);
            if (mul_mod(inverse,terms[0].value,prime)!=1) {
                PyMem_Free(terms);Py_DECREF(ops);
                PyErr_SetString(PyExc_ArithmeticError,"noninvertible modular pivot");goto fail;
            }
        }
        PyObject *reduced=PyDict_New();
        if (!reduced) {PyMem_Free(terms);Py_DECREF(ops);goto fail;}
        for (Py_ssize_t k=0;k<nnz;++k) {
            terms[k].value=mul_mod(terms[k].value,inverse,prime);
            PyObject *key_obj=PyLong_FromSsize_t(terms[k].col);
            PyObject *val_obj=PyLong_FromUnsignedLong(terms[k].value);
            if (!key_obj || !val_obj || PyDict_SetItem(reduced,key_obj,val_obj)<0) {
                Py_XDECREF(key_obj);Py_XDECREF(val_obj);Py_DECREF(reduced);
                Py_DECREF(ops);PyMem_Free(terms);goto fail;
            }
            Py_DECREF(key_obj);Py_DECREF(val_obj);
        }
        if (nnz) {
            pivots[i]=(Pivot){terms,nnz};column_pivot[terms[0].col]=i;
        }
        else PyMem_Free(terms);
        PyObject *inv=PyLong_FromUnsignedLong(inverse);
        PyObject *entry=inv ? Py_BuildValue("(OON)",reduced,ops,inv) : NULL;
        Py_DECREF(reduced);Py_DECREF(ops);
        if (!entry || PyList_Append(output,entry)<0) {Py_XDECREF(entry);goto fail;}
        Py_DECREF(entry);
    }
    goto finish;
fail:
    Py_CLEAR(output);
finish:
    sparse_free(&sparse);
    if (pivots) for (Py_ssize_t i=0;i<nrows;++i) PyMem_Free(pivots[i].terms);
    PyMem_Free(pivots);PyMem_Free(column_pivot);PyMem_Free(scratch);Py_XDECREF(rows);
    return output;
}

/* Shift sparse polynomial supports directly to matrix column IDs.  No
   temporary SymPy polynomial multiples are created in the compiled path. */
static PyObject *shifted_rows(PyObject *self, PyObject *args) {
    PyObject *basis, *specifications, *positions, *output=NULL;
    unsigned int prime;
    if (!PyArg_ParseTuple(args,"OOOI:shifted_rows",&basis,&specifications,&positions,&prime)) return NULL;
    if (!PyDict_Check(positions) || prime<2 || prime>(1U<<30)) {
        PyErr_SetString(PyExc_ValueError,"invalid shifted-row mapping or prime");return NULL;
    }
    PyObject *seq=PySequence_Fast(specifications,"reducer specs must be a sequence");
    PyObject *bases=PySequence_Fast(basis,"basis terms must be a sequence");
    if (!seq || !bases) {Py_XDECREF(seq);Py_XDECREF(bases);return NULL;}
    output=PyList_New(0);
    if (!output) goto error;
    for (Py_ssize_t i=0;i<native_sequence_size(seq);++i) {
        PyObject *spec=native_sequence_item(seq,i);
        if (!PyTuple_Check(spec) || PyTuple_Size(spec)!=2) {
            PyErr_SetString(PyExc_TypeError,"shift specification must be (powers, basis-index)");goto error;
        }
        PyObject *shift=PyTuple_GetItem(spec,0);
        Py_ssize_t index=PyLong_AsSsize_t(PyTuple_GetItem(spec,1));
        if (index == -1 && PyErr_Occurred()) goto error;
        if (index<0 || index>=native_sequence_size(bases) || !PyTuple_Check(shift)) {
            PyErr_SetString(PyExc_ValueError,"invalid shift or basis-index");goto error;
        }
        PyObject *terms=PySequence_Fast(native_sequence_item(bases,index),"basis row must be a term sequence");
        if (!terms) goto error;
        PyObject *row=PyDict_New();
        if (!row) {Py_DECREF(terms);goto error;}
        for (Py_ssize_t t=0;t<native_sequence_size(terms);++t) {
            PyObject *term=native_sequence_item(terms,t);
            if (!PyTuple_Check(term) || PyTuple_Size(term)!=2 || !PyTuple_Check(PyTuple_GetItem(term,0))) {
                PyErr_SetString(PyExc_TypeError,"term must be (exponents, coefficient)");Py_DECREF(row);Py_DECREF(terms);goto error;
            }
            PyObject *exp=PyTuple_GetItem(term,0);
            if (PyTuple_Size(exp)!=PyTuple_Size(shift)) {
                PyErr_SetString(PyExc_ValueError,"exponent length mismatch");Py_DECREF(row);Py_DECREF(terms);goto error;
            }
            PyObject *combined=PyTuple_New(PyTuple_Size(exp));
            if (!combined) {Py_DECREF(row);Py_DECREF(terms);goto error;}
            for (Py_ssize_t k=0;k<PyTuple_Size(exp);++k) {
                Py_ssize_t a=PyLong_AsSsize_t(PyTuple_GetItem(exp,k));
                Py_ssize_t b=PyLong_AsSsize_t(PyTuple_GetItem(shift,k));
                if (PyErr_Occurred() || a<0 || b<0 || a>PY_SSIZE_T_MAX-b) {
                    Py_DECREF(combined);Py_DECREF(row);Py_DECREF(terms);
                    if (!PyErr_Occurred())
                        PyErr_SetString(PyExc_ValueError,"invalid exponent shift");
                    goto error;
                }
                PyObject *val=PyLong_FromSsize_t(a+b);
                if (!val) {Py_DECREF(combined);Py_DECREF(row);Py_DECREF(terms);goto error;}
                PyTuple_SetItem(combined,k,val);
            }
            PyObject *column=PyDict_GetItemWithError(positions,combined);
            Py_DECREF(combined);
            if (!column) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_KeyError,"symbolic layout omitted shifted term");
                Py_DECREF(row);Py_DECREF(terms);goto error;
            }
            long long coefficient=PyLong_AsLongLong(PyTuple_GetItem(term,1));
            if (coefficient==-1 && PyErr_Occurred()) {Py_DECREF(row);Py_DECREF(terms);goto error;}
            long long residue=coefficient%(long long)prime;
            if (residue<0) residue+=prime;
            if (residue) {
                PyObject *value=PyLong_FromLongLong(residue);
                if (!value || PyDict_SetItem(row,column,value)<0) {
                    Py_XDECREF(value);Py_DECREF(row);Py_DECREF(terms);goto error;
                }
                Py_DECREF(value);
            }
        }
        Py_DECREF(terms);
        if (PyList_Append(output,row)<0) {Py_DECREF(row);goto error;}
        Py_DECREF(row);
    }
    Py_DECREF(seq);Py_DECREF(bases);return output;
error:
    Py_XDECREF(output);Py_DECREF(seq);Py_DECREF(bases);return NULL;
}

/* Exact multivariate division in GF(p) using sparse Python dictionaries but
   compiled term ordering and coefficient arithmetic. In particular, no
   Python-level SymPy Poly operations occur in the cancellation loop.
   Returned quotients satisfy f=sum(q[i]*g[i])+r modulo p, and every remainder
   monomial is indivisible by the leading monomials of g. */
static int nf_cmp(PyObject *a, PyObject *b, int order) {
    Py_ssize_t n=PyTuple_Size(a), d=0;
    for (Py_ssize_t i=0;i<n;++i) {
        Py_ssize_t x=PyLong_AsSsize_t(PyTuple_GetItem(a,i));
        Py_ssize_t y=PyLong_AsSsize_t(PyTuple_GetItem(b,i));
        if (PyErr_Occurred()) return 0;
        if (order && x!=y) d+=x-y;
        if (!order && x!=y) return x>y ? 1 : -1;
    }
    if (order && d) return d>0 ? 1 : -1;
    if (order==1) { /* grlex */
        for (Py_ssize_t i=0;i<n;++i) {
            Py_ssize_t x=PyLong_AsSsize_t(PyTuple_GetItem(a,i));
            Py_ssize_t y=PyLong_AsSsize_t(PyTuple_GetItem(b,i));
            if (x!=y) return x>y ? 1 : -1;
        }
    } else if (order==2) { /* grevlex */
        for (Py_ssize_t i=n;i>0;) {
            --i;
            Py_ssize_t x=PyLong_AsSsize_t(PyTuple_GetItem(a,i));
            Py_ssize_t y=PyLong_AsSsize_t(PyTuple_GetItem(b,i));
            if (x!=y) return x<y ? 1 : -1;
        }
    }
    return 0;
}
static int nf_validate_exps(PyObject *tuple, Py_ssize_t nvars) {
    if (!PyTuple_Check(tuple) || PyTuple_Size(tuple)!=nvars) return 0;
    for (Py_ssize_t k=0;k<nvars;++k) {
        Py_ssize_t power=PyLong_AsSsize_t(PyTuple_GetItem(tuple,k));
        if (power<0 || power>INT32_MAX || PyErr_Occurred()) return 0;
    }
    return 1;
}
static PyObject *nf_shift(PyObject *a, PyObject *b, int subtract) {
    Py_ssize_t n=PyTuple_Size(a);
    PyObject *out=PyTuple_New(n);
    if (!out) return NULL;
    for (Py_ssize_t i=0;i<n;++i) {
        Py_ssize_t x=PyLong_AsSsize_t(PyTuple_GetItem(a,i));
        Py_ssize_t y=PyLong_AsSsize_t(PyTuple_GetItem(b,i));
        if (PyErr_Occurred() || (subtract ? x<y : x>INT32_MAX-y)) {
            Py_DECREF(out);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_OverflowError,"invalid monomial shift");
            return NULL;
        }
        PyObject *z=PyLong_FromSsize_t(subtract ? x-y : x+y);
        if (!z) {Py_DECREF(out);return NULL;}
        PyTuple_SetItem(out,i,z);
    }
    return out;
}
static int nf_divisible(PyObject *large, PyObject *small) {
    for (Py_ssize_t i=0;i<PyTuple_Size(large);++i) {
        if (PyLong_AsSsize_t(PyTuple_GetItem(large,i)) <
            PyLong_AsSsize_t(PyTuple_GetItem(small,i))) return 0;
    }
    return 1;
}
static int nf_set(PyObject *dict, PyObject *key, uint32_t value) {
    if (!value) return PyDict_DelItem(dict,key);
    PyObject *obj=PyLong_FromUnsignedLong(value);
    if (!obj) return -1;
    int rc=PyDict_SetItem(dict,key,obj);
    Py_DECREF(obj);return rc;
}
/* Missing term means value zero; coefficient values are already reduced. */
static int nf_accumulate(PyObject *dict, PyObject *key, uint32_t delta, uint32_t p) {
    PyObject *old=PyDict_GetItemWithError(dict,key);
    if (!old && PyErr_Occurred()) return -1;
    uint32_t value=old ? (uint32_t)PyLong_AsUnsignedLong(old) : 0;
    if (PyErr_Occurred()) return -1;
    uint32_t result=(uint32_t)(((uint64_t)value+delta)%p);
    if (!result && !old) return 0;
    return nf_set(dict,key,result);
}
static PyObject *nf_normalize(PyObject *src, uint32_t p, Py_ssize_t *nvars, int *success) {
    PyObject *out=PyDict_New();
    if (!out) return NULL;
    Py_ssize_t pos=0;
    PyObject *key,*val;
    while (PyDict_Next(src,&pos,&key,&val)) {
        if (*nvars<0 && PyTuple_Check(key)) *nvars=PyTuple_Size(key);
        if (*nvars<0 || !nf_validate_exps(key,*nvars)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"invalid exponent tuple");
            Py_DECREF(out);return NULL;
        }
        long long v=PyLong_AsLongLong(val);
        if (PyErr_Occurred()) {Py_DECREF(out);return NULL;}
        long long mod=v%(long long)p;
        if (mod<0) mod+=p;
        if (mod && nf_set(out,key,(uint32_t)mod)<0) {Py_DECREF(out);return NULL;}
    }
    *success=1;
    return out;
}
static PyObject *normal_form(PyObject *self, PyObject *args) {
    PyObject *dividend,*divisor_input,*divisors=NULL,*current=NULL,*quotients=NULL,*remainder=NULL,*result=NULL;
    PyObject **basis=NULL,**lm=NULL;
    uint32_t *invs=NULL;
    unsigned int p;
    const char *order_name;
    Py_ssize_t max_steps=200000,max_terms=200000;
    int order=-1,ok=0;
    Py_ssize_t nvars=-1,nb=0;
    if (!PyArg_ParseTuple(args,"OOIs|nn:normal_form",&dividend,&divisor_input,&p,&order_name,&max_steps,&max_terms)) return NULL;
    if (!PyDict_Check(dividend) || p<2 || p>(1U<<30) || max_steps<1 || max_terms<1) {
        PyErr_SetString(PyExc_ValueError,"invalid GF(p) division arguments");return NULL;
    }
    if (!strcmp(order_name,"lex")) order=0;
    else if (!strcmp(order_name,"grlex")) order=1;
    else if (!strcmp(order_name,"grevlex")) order=2;
    else {PyErr_SetString(PyExc_ValueError,"order must be lex, grlex or grevlex");return NULL;}
    divisors=PySequence_Fast(divisor_input,"divisors must be a sequence");
    if (!divisors) return NULL;
    nb=native_sequence_size(divisors);
    basis=PyMem_Calloc((size_t)(nb?nb:1),sizeof(PyObject*));
    lm=PyMem_Calloc((size_t)(nb?nb:1),sizeof(PyObject*));
    invs=PyMem_Calloc((size_t)(nb?nb:1),sizeof(uint32_t));
    if (!basis || !lm || !invs) {PyErr_NoMemory();goto finish;}
    current=nf_normalize(dividend,p,&nvars,&ok);
    if (!current) goto finish;
    for (Py_ssize_t j=0;j<nb;++j) {
        PyObject *src=native_sequence_item(divisors,j);
        if (!PyDict_Check(src)) {PyErr_SetString(PyExc_TypeError,"divisor must be a dict");goto finish;}
        basis[j]=nf_normalize(src,p,&nvars,&ok);
        if (!basis[j]) goto finish;
        if (!PyDict_Size(basis[j])) {PyErr_SetString(PyExc_ZeroDivisionError,"zero polynomial divisor");goto finish;}
        Py_ssize_t pos=0; PyObject *key,*val,*best=NULL;
        while (PyDict_Next(basis[j],&pos,&key,&val)) {
            if (!best || nf_cmp(key,best,order)>0) best=key;
            if (PyErr_Occurred()) goto finish;
        }
        lm[j]=Py_NewRef(best);
        uint32_t lc=(uint32_t)PyLong_AsUnsignedLong(PyDict_GetItemWithError(basis[j],best));
        invs[j]=pow_mod(lc,p-2,p);
        if (!invs[j] || mul_mod(lc,invs[j],p)!=1) {
            PyErr_SetString(PyExc_ArithmeticError,"noninvertible divisor leading coefficient");goto finish;
        }
    }
    quotients=PyTuple_New(nb);
    if (!quotients) goto finish;
    for (Py_ssize_t j=0;j<nb;++j) {
        PyObject *dict=PyDict_New();
        if (!dict) goto finish;
        PyTuple_SetItem(quotients,j,dict);
    }
    remainder=PyDict_New();
    if (!remainder) goto finish;
    Py_ssize_t steps=0;
    while (PyDict_Size(current)) {
        if (steps++>=max_steps || PyDict_Size(current)>max_terms) {
            PyErr_SetString(PyExc_OverflowError,"GF(p) division resource limit");goto finish;
        }
        Py_ssize_t pos=0; PyObject *key,*val,*best=NULL;
        while (PyDict_Next(current,&pos,&key,&val)) {
            if (!best || nf_cmp(key,best,order)>0) best=key;
            if (PyErr_Occurred()) goto finish;
        }
        Py_INCREF(best);
        uint32_t coeff=(uint32_t)PyLong_AsUnsignedLong(PyDict_GetItemWithError(current,best));
        if (PyDict_DelItem(current,best)<0) {Py_DECREF(best);goto finish;}
        Py_ssize_t j=0;
        while (j<nb && !nf_divisible(best,lm[j])) ++j;
        if (j==nb) {
            if (nf_accumulate(remainder,best,coeff,p)<0) {Py_DECREF(best);goto finish;}
            if (PyDict_Size(remainder)>max_terms) {
                Py_DECREF(best);PyErr_SetString(PyExc_OverflowError,"GF(p) division resource limit");goto finish;
            }
        } else {
            PyObject *shift=nf_shift(best,lm[j],1);
            if (!shift) {Py_DECREF(best);goto finish;}
            uint32_t q=mul_mod(coeff,invs[j],p);
            if (nf_accumulate(PyTuple_GetItem(quotients,j),shift,q,p)<0) {Py_DECREF(shift);Py_DECREF(best);goto finish;}
            if (PyDict_Size(PyTuple_GetItem(quotients,j))>max_terms) {
                Py_DECREF(shift);Py_DECREF(best);PyErr_SetString(PyExc_OverflowError,"GF(p) division resource limit");goto finish;
            }
            Py_ssize_t dpos=0; PyObject *dkey,*dval;
            while (PyDict_Next(basis[j],&dpos,&dkey,&dval)) {
                if (dkey==lm[j] || PyObject_RichCompareBool(dkey,lm[j],Py_EQ)==1) continue;
                PyObject *sum=nf_shift(shift,dkey,0);
                if (!sum) {Py_DECREF(shift);Py_DECREF(best);goto finish;}
                uint32_t dc=(uint32_t)PyLong_AsUnsignedLong(dval);
                uint32_t sub=mul_mod(q,dc,p);
                if (nf_accumulate(current,sum,sub? p-sub:0,p)<0) {Py_DECREF(sum);Py_DECREF(shift);Py_DECREF(best);goto finish;}
                Py_DECREF(sum);
            }
            Py_DECREF(shift);
        }
        Py_DECREF(best);
    }
    result=PyTuple_Pack(2,quotients,remainder);
finish:
    Py_XDECREF(current);Py_XDECREF(quotients);Py_XDECREF(remainder);Py_XDECREF(divisors);
    if (basis) for (Py_ssize_t j=0;j<nb;++j) Py_XDECREF(basis[j]);
    if (lm) for (Py_ssize_t j=0;j<nb;++j) Py_XDECREF(lm[j]);
    PyMem_Free(basis);PyMem_Free(lm);PyMem_Free(invs);
    return result;
}

/* Batch S-polynomial shifted-support arithmetic entirely in compiled C.
   Pair descriptors contain (i,j,left_shift,right_shift,left_inverse,right_inverse).
   All multipliers and signs are supplied by the current-characteristic kernel.
   Returned dictionaries remain untrusted until exact rational replay. */
static PyObject *batched_spairs(PyObject *self, PyObject *args) {
    PyObject *basis_obj,*pairs_obj,*bases=NULL,*pairs=NULL,*out=NULL;
    unsigned int prime;
    if (!PyArg_ParseTuple(args,"OOI:batched_spairs",&basis_obj,&pairs_obj,&prime)) return NULL;
    if (prime<2 || prime>(1U<<30)) {PyErr_SetString(PyExc_ValueError,"invalid prime");return NULL;}
    bases=PySequence_Fast(basis_obj,"basis must be a sequence");
    pairs=PySequence_Fast(pairs_obj,"pairs must be a sequence");
    if (!bases || !pairs) goto done;
    out=PyList_New(0);
    if (!out) goto done;
    for (Py_ssize_t n=0;n<native_sequence_size(pairs);++n) {
        PyObject *desc=native_sequence_item(pairs,n);
        if (!PyTuple_Check(desc) || PyTuple_Size(desc)!=6) {
            PyErr_SetString(PyExc_TypeError,"pair descriptor must have six fields");goto error;
        }
        Py_ssize_t indices[2];
        indices[0]=PyLong_AsSsize_t(PyTuple_GetItem(desc,0));
        indices[1]=PyLong_AsSsize_t(PyTuple_GetItem(desc,1));
        if (PyErr_Occurred()) goto error;
        if (indices[0]<0 || indices[1]<0 ||
            indices[0]>=native_sequence_size(bases) ||
            indices[1]>=native_sequence_size(bases)) {
            PyErr_SetString(PyExc_IndexError,"invalid basis index");goto error;
        }
        PyObject *row=PyDict_New();
        if (!row) goto error;
        for (int which=0;which<2;++which) {
            PyObject *shift=PyTuple_GetItem(desc,2+which);
            unsigned long inverse=PyLong_AsUnsignedLong(PyTuple_GetItem(desc,4+which));
            if (!PyTuple_Check(shift) || !nf_validate_exps(shift,PyTuple_Size(shift)) ||
                PyErr_Occurred()) {
                if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError,"invalid shift exponents");
                Py_DECREF(row);goto error;
            }
            PyObject *poly=native_sequence_item(bases,indices[which]);
            if (!PyDict_Check(poly)) {
                PyErr_SetString(PyExc_TypeError,"basis polynomial must be a dict");
                Py_DECREF(row);goto error;
            }
            Py_ssize_t cursor=0;PyObject *powers,*coef;
            while (PyDict_Next(poly,&cursor,&powers,&coef)) {
                if (!nf_validate_exps(powers,PyTuple_Size(shift))) {
                    PyErr_SetString(PyExc_ValueError,"invalid exponent vector");
                    Py_DECREF(row);goto error;
                }
                PyObject *exp=nf_shift(powers,shift,0);
                if (!exp) {Py_DECREF(row);goto error;}
                long long c=PyLong_AsLongLong(coef);
                if (PyErr_Occurred()) {Py_DECREF(exp);Py_DECREF(row);goto error;}
                long long mod=c%(long long)prime;if (mod<0) mod+=prime;
                uint32_t product=mul_mod((uint32_t)mod,(uint32_t)(inverse%prime),prime);
                uint32_t delta=which ? (product ? prime-product:0) : product;
                if (nf_accumulate(row,exp,delta,prime)<0) {
                    Py_DECREF(exp);Py_DECREF(row);goto error;
                }
                Py_DECREF(exp);
            }
        }
        if (PyList_Append(out,row)<0) {Py_DECREF(row);goto error;}
        Py_DECREF(row);
    }
    goto done;
error:
    Py_CLEAR(out);
done:
    Py_XDECREF(bases);Py_XDECREF(pairs);
    return out;
}


/* Batched monomial arithmetic, independent of the coefficient field.
   Outputs are exact integer exponent vectors.  Input validation is mandatory
   because the CPython extension must never read invalid tuple offsets. */
static int mono_vector(PyObject *obj, Py_ssize_t width) {
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != width) return 0;
    PyObject *zero = PyLong_FromLong(0);
    if (!zero) return 0;
    for (Py_ssize_t i=0;i<width;i++) {
        PyObject *v=PyTuple_GetItem(obj,i);
        if (!PyLong_Check(v)) {Py_DECREF(zero);return 0;}
        int negative = PyObject_RichCompareBool(v,zero,Py_LT);
        if (negative != 0) {Py_DECREF(zero);return 0;}
    }
    Py_DECREF(zero);
    return 1;
}
static PyObject *mono_vectors(PyObject *arg, Py_ssize_t *width) {
    PyObject *rows=PySequence_Fast(arg,"expected monomial vectors");
    if (!rows) return NULL;
    for (Py_ssize_t i=0;i<native_sequence_size(rows);i++) {
        PyObject *v=native_sequence_item(rows,i);
        if (*width<0) {
            if (!PyTuple_Check(v)) {PyErr_SetString(PyExc_ValueError,"monomial must be a tuple");goto error;}
            *width=PyTuple_Size(v);
        }
        if (!mono_vector(v,*width)) {PyErr_SetString(PyExc_ValueError,"invalid monomial exponents");goto error;}
    }
    return rows;
error:
    Py_DECREF(rows);return NULL;
}
static PyObject *monomial_divisors(PyObject *self, PyObject *args) {
    PyObject *leader_arg,*term_arg,*leaders=NULL,*terms=NULL,*out=NULL;
    Py_ssize_t width=-1;
    if (!PyArg_ParseTuple(args,"OO:monomial_divisors",&leader_arg,&term_arg))return NULL;
    leaders=mono_vectors(leader_arg,&width);
    if (!leaders)goto done;
    terms=mono_vectors(term_arg,&width);
    if (!terms)goto done;
    out=PyTuple_New(native_sequence_size(terms));
    if (!out)goto done;
    for (Py_ssize_t j=0;j<native_sequence_size(terms);j++) {
        PyObject *term=native_sequence_item(terms,j);
        Py_ssize_t found=-1;
        for (Py_ssize_t i=0;i<native_sequence_size(leaders);i++) {
            PyObject *leader=native_sequence_item(leaders,i);
            int divides=1;
            for (Py_ssize_t k=0;k<width;k++) {
                int cmp=PyObject_RichCompareBool(PyTuple_GetItem(leader,k),PyTuple_GetItem(term,k),Py_LE);
                if (cmp<0)goto error;
                if (!cmp) {divides=0;break;}
            }
            if (divides) {found=i;break;}
        }
        PyObject *num=PyLong_FromSsize_t(found);
        if (!num)goto error;
        PyTuple_SetItem(out,j,num);
    }
    goto done;
error: Py_CLEAR(out);
done: Py_XDECREF(leaders);Py_XDECREF(terms);return out;
}
static PyObject *monomial_pairs(PyObject *self, PyObject *args) {
    PyObject *leader_arg,*pair_arg,*leaders=NULL,*pairs=NULL,*out=NULL;
    Py_ssize_t width=-1;
    if (!PyArg_ParseTuple(args,"OO:monomial_pairs",&leader_arg,&pair_arg))return NULL;
    leaders=mono_vectors(leader_arg,&width);
    if (!leaders)goto done;
    pairs=PySequence_Fast(pair_arg,"expected index pairs");
    if (!pairs)goto done;
    out=PyTuple_New(native_sequence_size(pairs));
    if (!out)goto done;
    for (Py_ssize_t q=0;q<native_sequence_size(pairs);q++) {
        PyObject *ij=native_sequence_item(pairs,q);
        if (!PyTuple_Check(ij) || PyTuple_Size(ij)!=2) {PyErr_SetString(PyExc_ValueError,"expected pair indices");goto error;}
        Py_ssize_t i=PyLong_AsSsize_t(PyTuple_GetItem(ij,0)),j=PyLong_AsSsize_t(PyTuple_GetItem(ij,1));
        if (PyErr_Occurred())goto error;
        if (i<0||j<0||i>=native_sequence_size(leaders)||j>=native_sequence_size(leaders)) {
            PyErr_SetString(PyExc_IndexError,"invalid pair indices");goto error;
        }
        PyObject *a=native_sequence_item(leaders,i),*b=native_sequence_item(leaders,j);
        PyObject *lcm=PyTuple_New(width);
        if (!lcm)goto error;
        int coprime=1;
        for (Py_ssize_t k=0;k<width;k++) {
            PyObject *u=PyTuple_GetItem(a,k),*v=PyTuple_GetItem(b,k);
            int cmp=PyObject_RichCompareBool(u,v,Py_GE);
            if (cmp<0) {Py_DECREF(lcm);goto error;}
            PyObject *max=cmp?u:v;Py_INCREF(max);PyTuple_SetItem(lcm,k,max);
            if (coprime) {
                int nz1=PyObject_IsTrue(u);
                /* avoid a leaked temporary and short-circuit through truth testing below */
                if (nz1<0) {Py_DECREF(lcm);goto error;}
                if (nz1) {
                    PyObject *zero=PyLong_FromLong(0);
                    if (!zero) {Py_DECREF(lcm);goto error;}
                    int nz2=PyObject_RichCompareBool(v,zero,Py_NE);Py_DECREF(zero);
                    if (nz2<0){Py_DECREF(lcm);goto error;}
                    if (nz2)coprime=0;
                }
            }
        }
        PyObject *item=PyTuple_Pack(2,lcm,coprime?Py_True:Py_False);Py_DECREF(lcm);
        if (!item)goto error;
        PyTuple_SetItem(out,q,item);
    }
    goto done;
error:Py_CLEAR(out);
done:Py_XDECREF(leaders);Py_XDECREF(pairs);return out;
}
static PyObject *monomial_sort(PyObject *self,PyObject *args) {
    PyObject *arg,*values=NULL,*keyed=NULL,*out=NULL;
    const char *order=NULL;int reverse=0;Py_ssize_t width=-1;
    if (!PyArg_ParseTuple(args,"Osp:monomial_sort",&arg,&order,&reverse))return NULL;
    if (strcmp(order,"lex") && strcmp(order,"grlex") && strcmp(order,"grevlex")) {
        PyErr_SetString(PyExc_ValueError,"unknown monomial order");return NULL;
    }
    values=mono_vectors(arg,&width);
    if (!values)goto done;
    keyed=PyList_New(0);
    if (!keyed)goto done;
    for (Py_ssize_t i=0;i<native_sequence_size(values);i++) {
        PyObject *vec=native_sequence_item(values,i),*key=NULL,*item=NULL;
        if (!strcmp(order,"lex")) {key=vec;Py_INCREF(key);}
        else {
            PyObject *sum=PyLong_FromLong(0),*second=PyTuple_New(width);
            if (!sum||!second) {Py_XDECREF(sum);Py_XDECREF(second);goto error;}
            for (Py_ssize_t k=0;k<width;k++) {
                PyObject *x=PyTuple_GetItem(vec,k);
                PyObject *next=PyNumber_Add(sum,x);Py_DECREF(sum);sum=next;
                if (!sum) {Py_DECREF(second);goto error;}
                PyObject *value;
                if (!strcmp(order,"grlex")) {value=x;Py_INCREF(value);}
                else value=PyNumber_Negative(PyTuple_GetItem(vec,width-k-1));
                if (!value) {Py_DECREF(sum);Py_DECREF(second);goto error;}
                PyTuple_SetItem(second,k,value);
            }
            key=PyTuple_Pack(2,sum,second);Py_DECREF(sum);Py_DECREF(second);
            if (!key)goto error;
        }
        item=PyTuple_Pack(2,key,vec);Py_DECREF(key);
        if (!item)goto error;
        if (PyList_Append(keyed,item)<0) {Py_DECREF(item);goto error;}
        Py_DECREF(item);
    }
    if (PyList_Sort(keyed)<0)goto error;
    out=PyTuple_New(PyList_Size(keyed));
    if (!out)goto error;
    for (Py_ssize_t i=0;i<PyList_Size(keyed);i++) {
        PyObject *pair=PyList_GetItem(keyed,reverse?PyList_Size(keyed)-i-1:i);
        PyObject *vec=PyTuple_GetItem(pair,1);Py_INCREF(vec);PyTuple_SetItem(out,i,vec);
    }
    goto done;
error:Py_CLEAR(out);
done:Py_XDECREF(values);Py_XDECREF(keyed);return out;
}

static PyMethodDef methods[] = {
    {"monomial_divisors",monomial_divisors,METH_VARARGS,"Batched exact first-divisor monomial queries."},
    {"monomial_pairs",monomial_pairs,METH_VARARGS,"Batched monomial LCM and coprimality."},
    {"monomial_sort",monomial_sort,METH_VARARGS,"Exact compiled monomial ordering."},
    {"batched_spairs",batched_spairs,METH_VARARGS,"Compiled exact GF(p) batched S-polynomial supports."},
    {"normal_form",normal_form,METH_VARARGS,"Exact sparse GF(p) polynomial division and quotients."},
    {"reduce_rows",reduce_rows,METH_VARARGS,"Adaptive sparse/dense GF(p) reducer with replayable operations."},
    {"shifted_rows",shifted_rows,METH_VARARGS,"Compiled symbolic polynomial-multiple assembly."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef definition={PyModuleDef_HEAD_INIT,"_ff_sparse",NULL,-1,methods};
PyMODINIT_FUNC PyInit__ff_sparse(void) {return PyModule_Create(&definition);}
