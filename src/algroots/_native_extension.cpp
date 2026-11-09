#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <flint/fq_nmod.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_factor.h>
#include <flint/ulong_extras.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <limits>
#include <memory>
namespace {
#include "_native_f4_python.h"

// This kernel owns FLINT coefficients and never invokes Python for row
// arithmetic. The prime-field kernel and its packed coefficients are unchanged.
struct FieldBudgetError : std::runtime_error {using std::runtime_error::runtime_error;};
using Exponents=std::vector<uint32_t>;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}

struct FieldContext {
 fq_nmod_ctx_t value;
 size_t degree;
 explicit FieldContext(const nmod_poly_t modulus):degree(nmod_poly_degree(modulus)){
  fq_nmod_ctx_init_modulus(value,modulus,"a");
 }
 ~FieldContext(){fq_nmod_ctx_clear(value);}
 FieldContext(const FieldContext&)=delete;
};

struct Coefficient {
 const FieldContext* field;
 fq_nmod_t value;
 explicit Coefficient(const FieldContext&ctx):field(&ctx){fq_nmod_init(value,field->value);}
 Coefficient(const Coefficient&other):Coefficient(*other.field){fq_nmod_set(value,other.value,field->value);}
 Coefficient(Coefficient&&other)noexcept:Coefficient(*other.field){fq_nmod_swap(value,other.value,field->value);}
 Coefficient& operator=(const Coefficient&other){fq_nmod_set(value,other.value,field->value);return *this;}
 Coefficient& operator=(Coefficient&&other)noexcept{fq_nmod_swap(value,other.value,field->value);return *this;}
 ~Coefficient(){fq_nmod_clear(value,field->value);}
 bool zero()const{return fq_nmod_is_zero(value,field->value);}
 bool operator==(const Coefficient&other)const{return fq_nmod_equal(value,other.value,field->value);}
};

struct MonomialOrder {
 int order;
 bool operator()(const Exponents&left,const Exponents&right)const {
  if(order){
   uint64_t a=std::accumulate(left.begin(),left.end(),uint64_t(0));
   uint64_t b=std::accumulate(right.begin(),right.end(),uint64_t(0));
   if(a!=b)return a>b;
  }
  if(order!=2)return left>right;
  for(size_t i=left.size();i-->0;)if(left[i]!=right[i])return left[i]<right[i];
  return false;
 }
};
using Polynomial=std::map<Exponents,Coefficient,MonomialOrder>;
static bool divides(const Exponents&a,const Exponents&b){
 for(size_t i=0;i<a.size();++i)if(a[i]>b[i])return false;
 return true;
}
static Exponents difference(const Exponents&a,const Exponents&b){
 Exponents result(a.size());for(size_t i=0;i<a.size();++i)result[i]=a[i]-b[i];return result;
}
static Exponents sum(const Exponents&a,const Exponents&b){
 Exponents result(a.size());for(size_t i=0;i<a.size();++i){
  if(a[i]>UINT32_MAX-b[i])throw FieldBudgetError("monomial exponent overflow");
  result[i]=a[i]+b[i];
 }return result;
}

struct ExtensionEngine {
 struct BatchMetric {size_t pairs,degree,rows,columns,before,after,max_degree;double symbolic,matrix,insertion;};
 const FieldContext&field;
 size_t n,max_pairs,max_terms,max_basis,batch_size,max_steps,max_cells;
 int order;
 std::vector<Polynomial> basis;
 std::multimap<uint64_t,std::tuple<size_t,size_t,Exponents>> pairs;
 size_t processed=0,steps=0,batches=0,peak_basis=0,max_rows=0,max_columns=0;
 double symbolic_seconds=0,matrix_seconds=0,insertion_seconds=0,finish_seconds=0;
 bool collect_metrics=false;
 std::vector<BatchMetric> metrics;
 ExtensionEngine(const FieldContext&f,size_t variables,int term_order,size_t mp,size_t mt,size_t mb,size_t bs,size_t ms,size_t mc)
  :field(f),n(variables),max_pairs(mp),max_terms(mt),max_basis(mb),batch_size(bs),max_steps(ms),max_cells(mc),order(term_order){}
 Polynomial polynomial()const{return Polynomial(MonomialOrder{order});}
 Coefficient one()const{Coefficient c(field);fq_nmod_one(c.value,field.value);return c;}
 void step(){if(++steps>max_steps)throw FieldBudgetError("finite-extension row-operation budget exceeded");if((steps&1023)==0)native_check_signals();}
 void bound(const Polynomial&poly)const{if(poly.size()>max_terms)throw FieldBudgetError("finite-extension polynomial term budget exceeded");}
 void subtract(Polynomial&target,const Polynomial&source,const Exponents&shift,const Coefficient&scale){
  Coefficient product(field);
  for(const auto&entry:source){
   step();fq_nmod_mul(product.value,scale.value,entry.second.value,field.value);
   Exponents monomial=sum(entry.first,shift);
   auto found=target.find(monomial);
   if(found==target.end()){
    fq_nmod_neg(product.value,product.value,field.value);
    if(!product.zero())target.emplace(std::move(monomial),product);
   }else{
    fq_nmod_sub(found->second.value,found->second.value,product.value,field.value);
    if(found->second.zero())target.erase(found);
   }bound(target);
  }
 }
 void monic(Polynomial&poly){
  if(poly.empty())return;
  Coefficient inverse(field);fq_nmod_inv(inverse.value,poly.begin()->second.value,field.value);
  for(auto&entry:poly){step();fq_nmod_mul(entry.second.value,entry.second.value,inverse.value,field.value);}
 }
 Polynomial normal_form(Polynomial poly,size_t skip=SIZE_MAX){
  Polynomial result=polynomial();
  while(!poly.empty()){
   native_check_signals();
   auto leading=poly.begin();size_t selected=SIZE_MAX;
   for(size_t i=0;i<basis.size();++i)if(i!=skip && !basis[i].empty() && divides(basis[i].begin()->first,leading->first)){selected=i;break;}
   if(selected==SIZE_MAX){result.emplace(leading->first,leading->second);poly.erase(leading);}
   else{
    Coefficient scale=leading->second;
    subtract(poly,basis[selected],difference(leading->first,basis[selected].begin()->first),scale);
   }
  }return result;
 }
 void insert(Polynomial poly){
  auto started=Clock::now();
  poly=normal_form(std::move(poly));
  if(poly.empty()){insertion_seconds+=seconds(started);return;}
  monic(poly);
  if(std::all_of(poly.begin()->first.begin(),poly.begin()->first.end(),[](uint32_t e){return e==0;})){
   basis.clear();pairs.clear();basis.push_back(std::move(poly));peak_basis=std::max(peak_basis,basis.size());insertion_seconds+=seconds(started);return;
  }
  if(basis.size()>=max_basis)throw FieldBudgetError("finite-extension basis budget exceeded");
  size_t index=basis.size();basis.push_back(std::move(poly));peak_basis=std::max(peak_basis,basis.size());
  for(size_t previous=0;previous<index;++previous){
   Exponents common(n);bool relatively_prime=true;
   for(size_t k=0;k<n;++k){
    uint32_t a=basis[index].begin()->first[k],b=basis[previous].begin()->first[k];
    common[k]=std::max(a,b);if(a&&b)relatively_prime=false;
   }
   if(relatively_prime)continue;
   if(pairs.size()>=max_pairs)throw FieldBudgetError("finite-extension queued-pair budget exceeded");
   uint64_t degree=std::accumulate(common.begin(),common.end(),uint64_t(0));
   pairs.emplace(degree,std::make_tuple(previous,index,std::move(common)));
  }insertion_seconds+=seconds(started);
 }
 Polynomial spoly(size_t i,size_t j,const Exponents&common){
  Polynomial result=polynomial();Coefficient negative=one();fq_nmod_neg(negative.value,negative.value,field.value);
  subtract(result,basis[i],difference(common,basis[i].begin()->first),negative);
  subtract(result,basis[j],difference(common,basis[j].begin()->first),one());return result;
 }
 void run(){
  while(!pairs.empty()){
   native_check_signals();++batches;
   size_t pair_start=processed,basis_start=basis.size();double insertion_start=insertion_seconds;
   uint64_t degree=pairs.begin()->first;std::vector<Polynomial> rows;
   while(!pairs.empty() && pairs.begin()->first==degree && processed-pair_start<batch_size){
    if(++processed>max_pairs)throw FieldBudgetError("finite-extension processed-pair budget exceeded");
    auto [i,j,common]=pairs.begin()->second;pairs.erase(pairs.begin());
    auto row=spoly(i,j,common);if(!row.empty())rows.push_back(std::move(row));
   }
   auto started=Clock::now();
   std::set<Exponents,MonomialOrder> pending(MonomialOrder{order}),columns(MonomialOrder{order});
   for(const auto&row:rows)for(const auto&entry:row)pending.insert(entry.first);
   std::set<std::pair<size_t,Exponents>> multiples;std::vector<Polynomial> reducers;
   size_t entries=0;for(const auto&row:rows)entries+=row.size();
   if(entries>max_terms)throw FieldBudgetError("finite-extension matrix entry budget exceeded");
   while(!pending.empty()){
    Exponents monomial=*pending.begin();pending.erase(pending.begin());
    if(!columns.insert(monomial).second)continue;
    if((columns.size()&511)==0)native_check_signals();
    if(columns.size()>max_terms)throw FieldBudgetError("finite-extension symbolic column budget exceeded");
    for(size_t i=0;i<basis.size();++i)if(divides(basis[i].begin()->first,monomial)){
     Exponents shift=difference(monomial,basis[i].begin()->first);
     if(multiples.emplace(i,shift).second){
      Polynomial row=polynomial();
      for(const auto&entry:basis[i]){Exponents m=sum(entry.first,shift);pending.insert(m);row.emplace(std::move(m),entry.second);}
      if(row.size()>max_terms-std::min(max_terms,entries))throw FieldBudgetError("finite-extension matrix entry budget exceeded");
      entries+=row.size();reducers.push_back(std::move(row));
     }break;
    }
   }
   size_t row_count=rows.size()+reducers.size();
   if(row_count && columns.size()>max_cells/row_count)throw FieldBudgetError("finite-extension matrix cell budget exceeded");
   double symbolic=seconds(started);
   max_rows=std::max(max_rows,row_count);max_columns=std::max(max_columns,columns.size());symbolic_seconds+=symbolic;
   started=Clock::now();
   std::map<Exponents,Polynomial,MonomialOrder> pivots(MonomialOrder{order});
   std::vector<Polynomial> candidates;
   auto eliminate=[&](Polynomial row,bool candidate){
    while(!row.empty()){
     auto pivot=pivots.find(row.begin()->first);
     if(pivot==pivots.end()){
      monic(row);Exponents leading=row.begin()->first;
      if(candidate)candidates.push_back(row);
      pivots.emplace(std::move(leading),std::move(row));return;
     }
     Coefficient scale=row.begin()->second;subtract(row,pivot->second,Exponents(n,0),scale);
    }
   };
   for(auto&row:reducers)eliminate(std::move(row),false);
   for(auto&row:rows)eliminate(std::move(row),true);
   double matrix=seconds(started);matrix_seconds+=matrix;
   // Echelon rows are not assumed reduced against later basis extensions.
   // Every candidate gets an exact normal form against the current basis.
   for(auto&row:candidates)insert(std::move(row));
   if(collect_metrics){
    size_t max_degree=0;for(const auto&monomial:columns)max_degree=std::max(max_degree,(size_t)std::accumulate(monomial.begin(),monomial.end(),uint64_t(0)));
    metrics.push_back({processed-pair_start,(size_t)degree,row_count,columns.size(),basis_start,basis.size(),max_degree,symbolic,matrix,insertion_seconds-insertion_start});
   }
  }
  auto started=Clock::now();bool changed=true;
  while(changed){
   changed=false;
   for(size_t i=0;i<basis.size();){
    auto reduced=normal_form(basis[i],i);monic(reduced);
    if(reduced.empty()){basis.erase(basis.begin()+i);changed=true;continue;}
    if(reduced!=basis[i]){basis[i]=std::move(reduced);changed=true;}++i;
   }
  }
  std::sort(basis.begin(),basis.end(),[&](const Polynomial&a,const Polynomial&b){return MonomialOrder{order}(a.begin()->first,b.begin()->first);});
  finish_seconds+=seconds(started);
 }
};

static Exponents read_monomial(PyObject*object,size_t n){
 auto sequence=own_python(PySequence_Fast(object,"monomial must be a sequence"));
 if(PySequence_Fast_GET_SIZE(sequence.get())!=(Py_ssize_t)n)throw std::invalid_argument("invalid monomial width");
 Exponents result(n);
 for(size_t i=0;i<n;++i){
  unsigned long long exponent=PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(sequence.get(),i));
  if(PyErr_Occurred()){preserve_python_interrupt();throw NativePythonError{};}
  if(exponent>UINT32_MAX)throw std::invalid_argument("exponent exceeds uint32 range");result[i]=(uint32_t)exponent;
 }return result;
}
static Coefficient read_coefficient(PyObject*object,const FieldContext&field,unsigned long prime){
 auto sequence=own_python(PySequence_Fast(object,"coefficient must be a sequence"));
 if(PySequence_Fast_GET_SIZE(sequence.get())>(Py_ssize_t)field.degree)throw std::invalid_argument("noncanonical extension coefficient");
 Coefficient result(field);
 for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(sequence.get());++i){
  unsigned long coefficient=PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(sequence.get(),i));
  if(PyErr_Occurred())throw NativePythonError{};nmod_poly_set_coeff_ui(result.value,i,coefficient%prime);
 }return result;
}
static PyObject* export_basis(const ExtensionEngine&engine){
 auto output=own_python(PyList_New(0));
 for(const auto&poly:engine.basis){
  auto row=own_python(PyDict_New());
  for(const auto&entry:poly){
   auto monomial=own_python(PyTuple_New(engine.n)),coefficient=own_python(PyTuple_New(engine.field.degree));
   for(size_t i=0;i<engine.n;++i)PyTuple_SET_ITEM(monomial.get(),i,own_python(PyLong_FromUnsignedLong(entry.first[i])).release());
   for(size_t i=0;i<engine.field.degree;++i)PyTuple_SET_ITEM(coefficient.get(),i,own_python(PyLong_FromUnsignedLong(nmod_poly_get_coeff_ui(entry.second.value,i))).release());
   check_python(PyDict_SetItem(row.get(),monomial.get(),coefficient.get()));
  }check_python(PyList_Append(output.get(),row.get()));
 }return output.release();
}
static PyObject* compute(PyObject*module,PyObject*args,PyObject*kwargs){
 PyObject *inputs,*modulus;unsigned long prime;int n;const char*order="grevlex";
 Py_ssize_t max_pairs=20000,max_terms=100000,max_basis=256,batch=8,max_steps=250000,max_degree=16,max_cells=4194304;
 int collect=0;
 static const char*names[]={"polynomials","prime","modulus","nvars","order","max_pairs","max_terms","max_basis","batch_size","max_reduction_steps","max_field_degree","max_matrix_cells","collect_batch_metrics",nullptr};
 if(!PyArg_ParseTupleAndKeywords(args,kwargs,"OkOi|snnnnnnnp",(char**)names,&inputs,&prime,&modulus,&n,&order,&max_pairs,&max_terms,&max_basis,&batch,&max_steps,&max_degree,&max_cells,&collect))return nullptr;
 try{
  if(prime<2||prime>1000000007||!n_is_prime(prime)||n<1||n>64)throw std::invalid_argument("expected a supported prime and 1..64 coordinates");
  if(max_pairs<1||max_terms<1||max_basis<1||batch<1||max_steps<1||max_degree<1||max_cells<1)throw std::invalid_argument("invalid finite-extension limits");
  int code=std::string(order)=="lex"?0:std::string(order)=="grlex"?1:std::string(order)=="grevlex"?2:-1;
  if(code<0)throw std::invalid_argument("unsupported monomial order");
  auto coefficients=own_python(PySequence_Fast(modulus,"modulus must be a sequence"));
  Py_ssize_t length=PySequence_Fast_GET_SIZE(coefficients.get());
  if(length<2)throw std::invalid_argument("extension modulus must have positive degree");
  if(length-1>max_degree)throw FieldBudgetError("extension modulus degree exceeds its budget");
  nmod_poly_t defining;nmod_poly_init(defining,prime);
  struct ModulusScope{nmod_poly_struct*value;~ModulusScope(){nmod_poly_clear(value);}}scope{defining};
  for(Py_ssize_t i=0;i<length;++i){
   unsigned long value=PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(coefficients.get(),i));
   if(PyErr_Occurred())throw NativePythonError{};nmod_poly_set_coeff_ui(defining,i,value%prime);
  }
  if(nmod_poly_degree(defining)!=length-1||nmod_poly_get_coeff_ui(defining,length-1)!=1||!nmod_poly_is_irreducible(defining))throw std::invalid_argument("extension modulus must be monic and irreducible");
  FieldContext field(defining);ExtensionEngine engine(field,n,code,max_pairs,max_terms,max_basis,batch,max_steps,max_cells);engine.collect_metrics=collect;
  auto sequence=own_python(PySequence_Fast(inputs,"polynomials must be a sequence"));std::vector<Polynomial> original;
  for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(sequence.get());++i){
   PyObject*row=PySequence_Fast_GET_ITEM(sequence.get(),i);
   if(!PyDict_Check(row))throw std::invalid_argument("polynomial must be a sparse dictionary");
   Polynomial polynomial=engine.polynomial();Py_ssize_t position=0;PyObject *key,*value;
   while(PyDict_Next(row,&position,&key,&value)){
    auto coefficient=read_coefficient(value,field,prime);
    if(!coefficient.zero())polynomial.emplace(read_monomial(key,n),std::move(coefficient));engine.bound(polynomial);
   }original.push_back(std::move(polynomial));
  }
  {NativeExecution execution;for(auto&poly:original)engine.insert(std::move(poly));engine.run();}
  auto basis=own_python(export_basis(engine));
  auto stats=own_python(Py_BuildValue("{s:s,s:s,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:d,s:d,s:d,s:d}",
   "algorithm","flint_sparse_f4","flint_version",flint_version,"field_degree",(Py_ssize_t)field.degree,
   "processed_pairs",(Py_ssize_t)engine.processed,"batches",(Py_ssize_t)engine.batches,"peak_basis_size",(Py_ssize_t)engine.peak_basis,
   "max_matrix_rows",(Py_ssize_t)engine.max_rows,"max_matrix_columns",(Py_ssize_t)engine.max_columns,"coefficient_operations",(Py_ssize_t)engine.steps,
   "symbolic_seconds",engine.symbolic_seconds,"matrix_seconds",engine.matrix_seconds,"insertion_seconds",engine.insertion_seconds,"finish_seconds",engine.finish_seconds));
  auto metrics=own_python(PyList_New(0));
  for(const auto&batch:engine.metrics){
   auto row=own_python(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:d,s:d,s:d}","pairs",(Py_ssize_t)batch.pairs,"lcm_degree",(Py_ssize_t)batch.degree,"matrix_rows",(Py_ssize_t)batch.rows,"matrix_columns",(Py_ssize_t)batch.columns,"basis_before",(Py_ssize_t)batch.before,"basis_after",(Py_ssize_t)batch.after,"max_term_degree",(Py_ssize_t)batch.max_degree,"symbolic_seconds",batch.symbolic,"matrix_seconds",batch.matrix,"insertion_seconds",batch.insertion));
   check_python(PyList_Append(metrics.get(),row.get()));
  }check_python(PyDict_SetItemString(stats.get(),"batch_metrics",metrics.get()));
  return PyTuple_Pack(2,basis.get(),stats.get());
 }catch(const NativePythonError&){return nullptr;}
 catch(const FieldBudgetError&error){
  PyObject*type=PyObject_GetAttrString(module,"LimitError");if(!type)return nullptr;PyErr_SetString(type,error.what());Py_DECREF(type);return nullptr;
 }catch(const std::invalid_argument&error){PyErr_SetString(PyExc_ValueError,error.what());return nullptr;}
 catch(const std::bad_alloc&){return PyErr_NoMemory();}
 catch(const std::exception&error){PyErr_SetString(PyExc_RuntimeError,error.what());return nullptr;}
}
static PyMethodDef methods[]={{"compute",(PyCFunction)compute,METH_VARARGS|METH_KEYWORDS,"Compute a reduced F4 basis over an explicitly defined finite extension field."},{nullptr,nullptr,0,nullptr}};
static PyModuleDef definition={PyModuleDef_HEAD_INIT,"_native_extension",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__native_extension(){
 try{
 auto module=own_python(PyModule_Create(&definition));
 auto error=own_python(PyErr_NewException("algroots._native_extension.LimitError",PyExc_RuntimeError,nullptr));
 if(PyModule_AddObject(module.get(),"LimitError",error.get())<0)return nullptr;error.release();
 if(PyModule_AddStringConstant(module.get(),"flint_version",flint_version)<0)return nullptr;
 return module.release();
 }catch(const NativePythonError&){return nullptr;}
 catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
