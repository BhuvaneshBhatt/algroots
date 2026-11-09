#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
struct NativeBudgetError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#include "_native_f4_python.h"

// Intern all monomial exponent vectors across one complete native computation.
// Immutable shared handles make polynomial, row and pair copies cheap.
struct MonHash {
  size_t operator()(const std::vector<unsigned> &a) const noexcept {
    size_t h = 1469598103934665603ULL;
    for (unsigned v : a) {
      h ^= v + 0x9e3779b9U;
      h *= 1099511628211ULL;
    }
    return h;
  }
};
// IDs are scoped to one native call: their only valid use is as cache keys
// within that computation.  Distinct prime computations cannot compare IDs.
struct InternedMon {
  std::shared_ptr<const std::vector<unsigned>> coefficients;
  uint32_t id;
  uint64_t total_degree;
  uint64_t support_mask;
  uint64_t packed_key;
  bool packed_valid;
};
struct MonPool {
  std::unordered_map<std::vector<unsigned>, InternedMon, MonHash> symbols;
  std::vector<InternedMon> by_id;
  size_t hits = 0, misses = 0, max_entries = 2000000;
  InternedMon intern(std::vector<unsigned> m) {
    if (((hits + misses) & 4095) == 4095)
      native_check_signals();
    auto found = symbols.find(m);
    if (found != symbols.end()) {
      ++hits;
      return found->second;
    }
    if (by_id.size() >= max_entries)
      throw NativeBudgetError("native monomial interning budget exceeded");
    auto coefficients =
        std::make_shared<const std::vector<unsigned>>(std::move(m));
    uint64_t degree = std::accumulate(coefficients->begin(),
                                      coefficients->end(), uint64_t{0});
    uint64_t support = 0, packed = 0;
    bool packable = coefficients->size() <= 8;
    for (size_t i = 0; i < coefficients->size(); ++i) {
      unsigned exponent = (*coefficients)[i];
      if (exponent && i < 64)
        support |= uint64_t{1} << i;
      if (exponent > 255)
        packable = false;
      packed = (packed << 8) | static_cast<uint8_t>(exponent);
    }
    InternedMon record{coefficients, static_cast<uint32_t>(by_id.size() + 1),
                       degree,       support,
                       packed,       packable};
    symbols.emplace(*coefficients, record);
    by_id.push_back(record);
    ++misses;
    return record;
  }
};
static thread_local MonPool *active_mon_pool = nullptr;
static constexpr bool active_packed_monomials = true;
class Mon {
  std::shared_ptr<const std::vector<unsigned>> data_;
  uint32_t intern_id_ = 0;
  uint64_t total_degree_ = 0;
  uint64_t support_mask_ = 0;
  uint64_t packed_key_ = 0;
  bool packed_valid_ = false;
  explicit Mon(int /*interned_only_tag*/) {}
  void assign(std::vector<unsigned> v) {
    if (active_mon_pool) {
      auto r = active_mon_pool->intern(std::move(v));
      data_ = std::move(r.coefficients);
      intern_id_ = r.id;
      total_degree_ = r.total_degree;
      support_mask_ = r.support_mask;
      packed_key_ = r.packed_key;
      packed_valid_ = r.packed_valid && active_packed_monomials;
    } else {
      total_degree_ = std::accumulate(v.begin(), v.end(), uint64_t{0});
      support_mask_ = 0;
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i] && i < 64)
          support_mask_ |= (uint64_t{1} << i);
      packed_valid_ = active_packed_monomials && v.size() <= 8;
      packed_key_ = 0;
      for (unsigned exp : v) {
        if (exp > 255)
          packed_valid_ = false;
        packed_key_ = (packed_key_ << 8) | static_cast<uint8_t>(exp);
      }
      data_ = std::make_shared<const std::vector<unsigned>>(std::move(v));
    }
  }

public:
  Mon() { assign({}); }

  static Mon from_id(uint32_t id) {
    if (!active_mon_pool || id == 0 || id > active_mon_pool->by_id.size())
      throw std::runtime_error("invalid contiguous monomial ID");
    const auto &r = active_mon_pool->by_id[id - 1];
    Mon m(0);
    m.data_ = r.coefficients;
    m.intern_id_ = r.id;
    m.total_degree_ = r.total_degree;
    m.support_mask_ = r.support_mask;
    m.packed_key_ = r.packed_key;
    m.packed_valid_ = r.packed_valid && active_packed_monomials;
    return m;
  }
  explicit Mon(std::vector<unsigned> v) { assign(std::move(v)); }
  Mon(size_t n, unsigned value) { assign(std::vector<unsigned>(n, value)); }
  Mon(std::initializer_list<unsigned> v) { assign(std::vector<unsigned>(v)); }
  size_t size() const { return data_->size(); }
  bool empty() const { return data_->empty(); }
  unsigned operator[](size_t i) const { return (*data_)[i]; }
  auto begin() const { return data_->begin(); }
  auto end() const { return data_->end(); }
  const std::vector<unsigned> &values() const { return *data_; }
  uint32_t id() const { return intern_id_; }
  uint64_t total_degree() const { return total_degree_; }
  uint64_t support_mask() const { return support_mask_; }
  bool packed_valid() const { return packed_valid_; }
  uint64_t packed_key() const { return packed_key_; }
  bool operator<(const Mon &o) const {
    if (size() == o.size() && packed_valid_ && o.packed_valid_)
      return packed_key_ < o.packed_key_;
    return *data_ < *o.data_;
  }
  bool operator==(const Mon &o) const {
    if (size() == o.size() && packed_valid_ && o.packed_valid_)
      return packed_key_ == o.packed_key_;
    return (intern_id_ && intern_id_ == o.intern_id_ && data_ == o.data_) ||
           data_ == o.data_ || *data_ == *o.data_;
  }
  bool operator!=(const Mon &o) const { return !(*this == o); }
};
struct MonPoolScope {
  MonPool *before;
  explicit MonPoolScope(MonPool &pool) : before(active_mon_pool) {
    active_mon_pool = &pool;
  }
  ~MonPoolScope() { active_mon_pool = before; }
};
static int compare(const Mon &a, const Mon &b, int order);
// Polynomial maps capture the term order so their first term is the lead.
static thread_local int active_poly_order = 2;
struct MonOrder {
  int order = active_poly_order;
  bool operator()(const Mon &a, const Mon &b) const {
    return compare(a, b, order) > 0;
  }
};
using Poly = std::map<Mon, uint64_t, MonOrder>;

// 8-byte flat term: globally interned monomial ID plus GF(p) coefficient.
// Consecutive offsets delimit the polynomials in one contiguous term arena.
struct FlatTerm {
  uint32_t monomial;
  uint32_t coefficient;
};
static_assert(sizeof(FlatTerm) == 8, "flat polynomial terms must be 8 bytes");
using Pair = std::tuple<size_t, size_t, Mon>;
using Reducer = std::pair<Mon, size_t>;
struct Layout {
  std::vector<Mon> columns;
  std::vector<Reducer> reducers;
};
struct Engine {
  unsigned n, p;
  int order;
  size_t max_pairs, max_terms, max_basis, batch;
  std::vector<Poly> basis;
  size_t max_queued_pairs = 0;
  // Degree-indexed FIFO preserves increasing LCM-degree selection without
  // rescanning the entire pending pair list for every batch.

  std::map<uint64_t, std::deque<Pair>> f4_degree_queue;
  size_t f4_degree_queue_size = 0;
  size_t f4_pair_select_ns = 0, f4_spoly_ns = 0, f4_symbolic_ns = 0;
  size_t f4_assembly_ns = 0, f4_elimination_ns = 0, f4_insertion_ns = 0,
         f4_finish_ns = 0;
  size_t f4_index_lookup_hits = 0;

  bool collect_batch_metrics = false;

  // Accepted matrices: pairs, LCM min/max, rows, columns,
  // basis before/after, symbolic/matrix/insertion ns, max term degree.
  std::vector<std::array<size_t, 11>> batch_metrics;

  size_t peak_intermediate_basis = 0, pre_interreduction_basis = 0;

  size_t finish_pruned_lm = 0, finish_skipped_nf = 0, finish_nf_calls = 0;
  size_t matrix_tile_sparse = 0, matrix_tile_dense = 0, matrix_tile_updates = 0,
         matrix_tiled_rows = 0;
  size_t block_reducer_rows = 0, block_work_rows = 0;
  size_t block_reducer_nnz = 0, block_work_nnz = 0;
  size_t block_reducer_cells = 0, block_work_cells = 0;
  size_t block_work_ge20pct = 0, block_work_ge40pct = 0;
  size_t block_sparse_rows = 0, block_dense_rows = 0;
  size_t block_reducer_pivots = 0, block_work_pivots = 0;

  size_t matrix_peak_width = 0, matrix_peak_rows = 0,
         matrix_peak_reducer_rows = 0;
  size_t matrix_peak_echelon = 0, matrix_preprocessed_batches = 0;

  size_t pair_index_candidates = 0, pair_index_degree_skips = 0;
  size_t symbolic_hash_columns = 0, symbolic_hash_batches = 0;
  size_t prospective_chain_pairs = 0, prospective_checks = 0;
  size_t f4_reduction_passes = 0;
  std::vector<FlatTerm> flat_basis_terms;
  std::vector<size_t> flat_basis_offsets{0};
  size_t flat_matrix_terms = 0, flat_pivot_terms_peak = 0,
         flat_row_reductions = 0, flat_matrix_batches = 0;
  size_t flat_reducer_pivots_excluded = 0;
  size_t f4_reducer_candidates = 0, f4_reducer_divides = 0;

  size_t layout_rows = 0, layout_columns = 0, emitted_rows = 0;

  size_t chain_pairs_pruned = 0;

  size_t pairs = 0, batches = 0;
  mutable size_t nf_steps = 0;
  size_t max_nf_steps = 250000;
  bool step_limit() const {
    ++nf_steps;
    if ((nf_steps & 1023) == 0)
      native_check_signals();
    return nf_steps > max_nf_steps;
  }
};
static Mon plus(const Mon &a, const Mon &b) {

  std::vector<unsigned> v(a.values());
  for (size_t i = 0; i < v.size(); ++i) {
    if (b[i] > UINT32_MAX - v[i])
      throw std::runtime_error("monomial exponent overflow");
    v[i] += b[i];
  }
  return Mon(std::move(v));
}
static Mon minus(const Mon &a, const Mon &b) {

  std::vector<unsigned> v(a.values());
  for (size_t i = 0; i < v.size(); ++i)
    v[i] -= b[i];
  return Mon(std::move(v));
}
static Mon mlcm(const Mon &a, const Mon &b) {

  std::vector<unsigned> v(a.values());
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = std::max(v[i], b[i]);
  return Mon(std::move(v));
}
struct MonomialDivisibility {
  static inline bool exact(const Mon &a, const Mon &b) {
    if (a.total_degree() > b.total_degree() ||
        (a.support_mask() & ~b.support_mask()))
      return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] > b[i])
        return false;
    return true;
  }
};
static inline bool divides(const Mon &a, const Mon &b) {
  return MonomialDivisibility::exact(a, b);
}

struct ReducerSelection {
  template <class Candidates, class Eligible>
  static inline size_t first(const Candidates &candidates,
                             Eligible &&eligible) {
    for (size_t pos = 0; pos < candidates.size(); ++pos) {
      const size_t i = candidates[pos];
      if (eligible(i))
        return i;
    }
    return SIZE_MAX;
  }
};
struct DenseReducerCandidates {
  size_t count;
  size_t size() const { return count; }
  size_t operator[](size_t i) const { return i; }
};

static uint64_t degree(const Mon &a) { return a.total_degree(); }
static int compare(const Mon &a, const Mon &b, int order) {
  if (order) {
    uint64_t da = degree(a), db = degree(b);
    if (da != db)
      return da > db ? 1 : -1;
  }
  if (a.size() == b.size() && a.packed_valid() && b.packed_valid()) {
    const uint64_t x = a.packed_key() ^ b.packed_key();
    if (!x)
      return 0;
    if (order != 2)
      return a.packed_key() > b.packed_key() ? 1 : -1;
    // Most significant variable occupies the high byte. Grevlex compares
    // the rightmost different exponent with the REVERSED sense.
#if defined(_MSC_VER)
    unsigned long first_bit;
    _BitScanForward64(&first_bit, x);
    const unsigned shift = (static_cast<unsigned>(first_bit) >> 3) * 8;
#else
    const unsigned shift = (static_cast<unsigned>(__builtin_ctzll(x)) >> 3) * 8;
#endif
    return ((a.packed_key() >> shift) & 255) < ((b.packed_key() >> shift) & 255)
               ? 1
               : -1;
  }
  if (order == 2) {
    for (size_t k = a.size(); k-- > 0;)
      if (a[k] != b[k])
        return a[k] < b[k] ? 1 : -1;
  } else
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i])
        return a[i] > b[i] ? 1 : -1;
  return 0;
}
static Mon lead(const Poly &f, const Engine &) {
  if (f.empty())
    throw std::runtime_error("zero leading monomial");
  // Every Poly map has a comparator fixed to the term order at creation.
  return f.begin()->first;
}
static uint64_t inverse(uint64_t a, unsigned p) {
  if (!a)
    throw std::runtime_error("division by zero");
  int64_t t = 0, b = 1, r = p, s = a;
  while (s) {
    int64_t q = r / s;
    auto x = t - q * b;
    t = b;
    b = x;
    auto y = r - q * s;
    r = s;
    s = y;
  }
  if (r != 1)
    throw std::runtime_error("modulus not prime");
  return (t % p + p) % p;
}
template <class Storage> struct SparseArithmetic;
template <> struct SparseArithmetic<Poly> {
  static inline void subtract_shifted(Poly &f, const Poly &g, const Mon &shift,
                                      uint64_t scale, const Engine &e) {
    for (const auto &[m, c] : g) {
      Mon k = plus(m, shift);
      uint64_t a = c * scale % e.p;
      auto it = f.find(k);
      uint64_t old = it == f.end() ? 0 : it->second;
      uint64_t next = (old + e.p - a) % e.p;
      if (next)
        f[std::move(k)] = next;
      else if (it != f.end())
        f.erase(it);
    }
    if (f.size() > e.max_terms)
      throw NativeBudgetError("native F4 term budget exceeded");
  }
};
static inline void subtract_multiple(Poly &f, const Poly &g, const Mon &shift,
                                     uint64_t scale, const Engine &e) {
  SparseArithmetic<Poly>::subtract_shifted(f, g, shift, scale, e);
}
static void monic(Poly &f, const Engine &e) {
  if (f.empty())
    return;
  uint64_t scalar = inverse(f.begin()->second, e.p);
  for (auto &[k, c] : f)
    c = c * scalar % e.p;
}

static Poly normal_form(Poly f, const std::vector<Poly> &divisors,
                        const Engine &e, size_t skip = SIZE_MAX) {
  Poly remainder;
  // Cache leading monomials once per normal-form call. Re-scanning full
  // reducer polynomials for each intermediate term is a major F4 bottleneck.
  std::vector<Mon> divisor_leads;
  divisor_leads.reserve(divisors.size());
  for (const auto &d : divisors)
    divisor_leads.push_back(d.empty() ? Mon{} : lead(d, e));
  while (!f.empty()) {
    if (e.step_limit())
      throw NativeBudgetError("native F4 normal-form step budget exceeded");
    Mon m = f.begin()->first;
    uint64_t c = f.begin()->second;
    bool done = false;
    for (size_t i = 0; i < divisors.size(); ++i) {
      if (i == skip)
        continue;
      const auto &d = divisors[i];
      if (d.empty())
        continue;
      const Mon &lm = divisor_leads[i];
      if (!divides(lm, m))
        continue;

      subtract_multiple(f, d, minus(m, lm), c * inverse(d.at(lm), e.p) % e.p,
                        e);
      done = true;
      break;
    }
    if (!done) {
      remainder.emplace(m, c);
      f.erase(f.begin());
    }
    if (remainder.size() > e.max_terms)
      throw NativeBudgetError("native F4 remainder budget exceeded");
  }
  return remainder;
}
// Work only with interned monomial IDs and a sparse hash coefficient table.
// The priority queue is a lazy ordered frontier: polynomial reductions only
// produce monomials strictly smaller than the current leading monomial.
// We verify the same strict descent as the ordered-map normal form.
static Poly hashed_normal_form(Poly f, const std::vector<Poly> &divisors,
                               const Engine &e, size_t skip = SIZE_MAX) {
  if (!active_mon_pool)
    throw std::runtime_error("hashed normal form requires monomial interning");

  struct LowerPriority {
    int order;
    bool operator()(const Mon &a, const Mon &b) const {
      return compare(a, b, order) < 0;
    }
  };
  std::priority_queue<Mon, std::vector<Mon>, LowerPriority> frontier(
      LowerPriority{e.order});
  std::unordered_map<uint32_t, uint32_t> coefficients;
  coefficients.reserve(f.size() * 2 + 16);
  for (const auto &[m, c] : f) {
    if (!m.id())
      throw std::runtime_error("uninterned normal-form monomial");
    coefficients.emplace(m.id(), static_cast<uint32_t>(c));
    frontier.push(m);
  }
  std::vector<FlatTerm> remainder;
  std::vector<Mon> leading;
  leading.reserve(divisors.size());
  std::vector<uint64_t> inverse_leads;
  inverse_leads.reserve(divisors.size());
  for (const auto &g : divisors) {
    if (g.empty()) {
      leading.emplace_back();
      inverse_leads.push_back(0);
    } else {
      Mon m = lead(g, e);
      leading.push_back(m);
      inverse_leads.push_back(inverse(g.begin()->second, e.p));
    }
  }
  while (!frontier.empty()) {
    Mon m = frontier.top();
    frontier.pop();
    auto found = coefficients.find(m.id());
    if (found == coefficients.end())
      continue;
    const uint32_t c = found->second;
    if (e.step_limit())
      throw NativeBudgetError("native F4 normal-form step budget exceeded");
    size_t chosen = divisors.size();
    for (size_t i = 0; i < divisors.size(); ++i) {
      if ((i != skip) && inverse_leads[i] && divides(leading[i], m)) {
        chosen = i;
        break;
      }
    }
    coefficients.erase(found);
    if (chosen == divisors.size()) {
      remainder.push_back({m.id(), c});
      if (remainder.size() > e.max_terms)
        throw NativeBudgetError("native F4 remainder budget exceeded");
      continue;
    }

    const Mon shift = minus(m, leading[chosen]);
    const uint64_t scale =
        static_cast<uint64_t>(c) * inverse_leads[chosen] % e.p;
    for (const auto &[term, coefficient] : divisors[chosen]) {
      if (term == leading[chosen])
        continue; // leading term cancels exactly
      const Mon shifted = plus(term, shift);
      if (compare(shifted, m, e.order) >= 0)
        throw std::runtime_error("nondecreasing hashed polynomial reduction");
      const auto previous = coefficients.find(shifted.id());
      const uint32_t old =
          previous == coefficients.end() ? 0 : previous->second;
      const uint32_t amount = static_cast<uint32_t>(scale * coefficient % e.p);
      const uint32_t next = static_cast<uint32_t>(
          (static_cast<uint64_t>(old) + e.p - amount) % e.p);
      if (!next)
        coefficients.erase(shifted.id());
      else {
        if (!old)
          frontier.push(shifted);
        coefficients[shifted.id()] = next;
      }
    }
    if (coefficients.size() > e.max_terms)
      throw NativeBudgetError("native F4 term budget exceeded");
  }
  Poly result;
  for (const auto &term : remainder)
    result.emplace(Mon::from_id(term.monomial), term.coefficient);
  return result;
}

static uint64_t f4_elapsed_ns(std::chrono::steady_clock::time_point start);

static void insert(Engine &e, Poly f) {
  if (f.empty())
    return;

  // Reduce against the entire current basis: earlier insertions can introduce
  // new reducers after the matrix echelon rows were computed.

  f = ((e.basis.size() > 64))
          ? hashed_normal_form(std::move(f), e.basis, e, SIZE_MAX)
          : normal_form(std::move(f), e.basis, e, SIZE_MAX);

  if (f.empty())
    return;
  monic(f, e);
  Mon lm = lead(f, e);

  // Prospective Buchberger chain criterion. Both replacement obligations
  // have strictly smaller LCM monomials and are therefore processed earlier
  // in the degree-indexed pair queue; equal-LCM cases are never discarded.
  std::vector<Mon> old_leads, new_lcms;
  {
    old_leads.reserve(e.basis.size());
    new_lcms.reserve(e.basis.size());
    for (const auto &g : e.basis) {
      old_leads.push_back(lead(g, e));
      new_lcms.push_back(mlcm(old_leads.back(), lm));
    }
  }
  std::vector<size_t> degree_candidates;
  const bool use_indexed_chain = (((e.basis.size() > 128)));
  if ((use_indexed_chain)) {
    degree_candidates.resize(new_lcms.size());
    std::iota(degree_candidates.begin(), degree_candidates.end(), 0);
    std::sort(degree_candidates.begin(), degree_candidates.end(),
              [&](size_t a, size_t b) {
                if (degree(new_lcms[a]) != degree(new_lcms[b]))
                  return degree(new_lcms[a]) < degree(new_lcms[b]);
                return a < b;
              });
  }
  for (size_t i = 0; i < e.basis.size(); ++i) {
    Mon old = (old_leads[i]);
    Mon l = (new_lcms[i]);
    if (degree(l) == degree(old) + degree(lm))
      continue; // Buchberger product criterion
    {
      bool redundant = false;
      if (use_indexed_chain) {
        const auto stop =
            std::lower_bound(degree_candidates.begin(), degree_candidates.end(),
                             degree(l), [&](size_t k, uint64_t bound) {
                               return degree(new_lcms[k]) < bound;
                             });
        e.pair_index_degree_skips += degree_candidates.end() - stop;
        for (auto it = degree_candidates.begin(); it != stop; ++it) {
          size_t k = *it;
          if (i == k)
            continue;
          const auto &candidate = new_lcms[k];
          if (candidate.support_mask() & ~l.support_mask())
            continue;
          ++e.pair_index_candidates;
          ++e.prospective_checks;
          if (!divides(candidate, l))
            continue;
          if (mlcm(old, old_leads[k]) != l) {
            redundant = true;
            break;
          }
        }
      } else {
        for (size_t k = 0; k < e.basis.size(); ++k) {
          if (i == k)
            continue;
          ++e.prospective_checks;
          if (degree(new_lcms[k]) >= degree(l) || !divides(new_lcms[k], l))
            continue;
          // lcm(old_i,old_k) divides l; require a *strict* divisor.
          if (mlcm(old, old_leads[k]) != l) {
            redundant = true;
            break;
          }
        }
      }
      if (redundant) {
        ++e.prospective_chain_pairs;
        continue;
      }
    }
    {
      uint64_t d = (degree(l));

      e.f4_degree_queue[d].emplace_back(i, e.basis.size(), std::move(l));
      ++e.f4_degree_queue_size;
    }
    e.max_queued_pairs = std::max(e.max_queued_pairs, (e.f4_degree_queue_size));
  }
  // Buchberger's LCM-chain criterion: old pair (i,j) is redundant if
  // the new leading monomial connects i and j through two strictly
  // smaller LCMs. Both new pairs are queued or satisfy the product
  // criterion, so the corresponding S-polynomial representation follows
  // from syzygies with strictly smaller critical monomials.
  {
    for (auto bucket = e.f4_degree_queue.begin();
         bucket != e.f4_degree_queue.end();) {
      auto &entries = bucket->second;
      for (auto it = entries.begin(); it != entries.end();) {
        size_t i = std::get<0>(*it), j = std::get<1>(*it);
        if (j == e.basis.size() || !divides(lm, std::get<2>(*it))) {
          ++it;
          continue;
        }
        Mon a = mlcm(lm, lead(e.basis[i], e));
        Mon b = mlcm(lm, lead(e.basis[j], e));
        if (a != std::get<2>(*it) && b != std::get<2>(*it)) {
          it = entries.erase(it);
          ++e.chain_pairs_pruned;
          --e.f4_degree_queue_size;
        } else
          ++it;
      }
      if (entries.empty())
        bucket = e.f4_degree_queue.erase(bucket);
      else
        ++bucket;
    }
  }

  {
    if (!active_mon_pool)
      throw std::runtime_error("contiguous basis requires monomial interning");
    for (const auto &[mon, c] : f) {
      if (!mon.id() || c > UINT32_MAX)
        throw std::runtime_error("invalid flat basis term");
      e.flat_basis_terms.push_back({mon.id(), static_cast<uint32_t>(c)});
    }
    e.flat_basis_offsets.push_back(e.flat_basis_terms.size());
  }

  e.basis.push_back(std::move(f));

  e.peak_intermediate_basis =
      std::max(e.peak_intermediate_basis, e.basis.size());
  if (e.basis.size() > e.max_basis)
    throw NativeBudgetError("native F4 basis limit");
}
static Poly s_poly(const Poly &a, const Poly &b, const Mon &l,
                   const Engine &e) {
  Mon lm = lead(a, e), rm = lead(b, e);
  Poly s;
  subtract_multiple(s, a, minus(l, lm), e.p - inverse(a.at(lm), e.p), e);
  subtract_multiple(s, b, minus(l, rm), inverse(b.at(rm), e.p), e);
  return s;
}
// Merge shifted sorted streams: translation preserves the term order,
// and the two leading terms cancel before lower coefficients are combined.
static Poly merged_s_poly(const Poly &a, const Poly &b, const Mon &l,
                          const Engine &e) {
  const Mon shift_a = minus(l, lead(a, e)), shift_b = minus(l, lead(b, e));
  const uint64_t inv_a = inverse(a.begin()->second, e.p),
                 inv_b = inverse(b.begin()->second, e.p);
  auto ai = std::next(a.begin()), bi = std::next(b.begin());
  std::optional<Mon> ma, mb;
  Poly result;
  while (ai != a.end() || bi != b.end()) {
    if (ai != a.end() && !ma)
      ma.emplace(plus(ai->first, shift_a));
    if (bi != b.end() && !mb)
      mb.emplace(plus(bi->first, shift_b));
    const int comparison = !ma ? -1 : !mb ? 1 : compare(*ma, *mb, e.order);
    uint64_t value = 0;
    if (comparison >= 0)
      value = ai->second * inv_a % e.p;
    if (comparison <= 0)
      value = (value + e.p - bi->second * inv_b % e.p) % e.p;
    if (value)
      result.emplace_hint(result.end(), comparison >= 0 ? *ma : *mb, value);
    if (comparison >= 0) {
      ++ai;
      ma.reset();
    }
    if (comparison <= 0) {
      ++bi;
      mb.reset();
    }
  }
  if (result.size() > e.max_terms)
    throw NativeBudgetError("native F4 term budget exceeded");
  return result;
}

static Layout layout(Engine &e, const std::vector<Poly> &ss) {

  std::set<Mon> supports, visited;

  auto insert_support = [&](const Mon &candidate) {
    bool inserted = false;
    inserted = supports.insert(candidate).second;

    return inserted;
  };
  auto support_size = [&]() { return (supports.size()); };
  std::map<Mon, unsigned> pending;
  std::vector<Mon> leading;
  for (const auto &f : e.basis)
    leading.push_back(lead(f, e));
  for (const auto &f : ss)
    for (const auto &[m, c] : f) {
      insert_support(m);
      pending[m] = 0;
    }
  std::vector<Reducer> reducers;
  while (!pending.empty()) {
    auto selected = pending.begin();
    Mon m = selected->first;
    unsigned depth = selected->second;
    pending.erase(selected);
    if (!visited.insert(m).second)
      continue;

    auto eligible = [&](size_t i) {
      ++e.f4_reducer_candidates;
      if (!divides(leading[i], m))
        return false;
      ++e.f4_reducer_divides;
      return true;
    };
    size_t chosen = SIZE_MAX;

    if (chosen == SIZE_MAX) {
      chosen = (ReducerSelection::first(DenseReducerCandidates{e.basis.size()},
                                        eligible));
    }
    if (chosen != SIZE_MAX) {
      const Mon shift = minus(m, leading[chosen]);
      reducers.emplace_back(shift, chosen);

      auto enqueue = [&](Mon new_m) {
        const bool inserted = insert_support(new_m);

        if (inserted)
          pending.emplace(std::move(new_m), depth + 1);
      };
      {
        const auto &poly = e.basis[chosen];
        for (auto it = poly.begin(); it != poly.end(); ++it) {

          enqueue(plus(it->first, shift));
        }
      }
    }
    if (support_size() > e.max_terms || reducers.size() > e.max_terms)
      throw NativeBudgetError("native F4 symbolic budget exceeded");
  }
  std::vector<Mon> cols;
  cols.assign(supports.begin(), supports.end());
  std::sort(cols.begin(), cols.end(), [&](const auto &a, const auto &b) {
    return compare(a, b, e.order) > 0;
  });
  std::sort(reducers.begin(), reducers.end(),
            [&](const auto &a, const auto &b) {
              return compare(plus(a.first, leading[a.second]),
                             plus(b.first, leading[b.second]), e.order) > 0;
            });

  return {std::move(cols), std::move(reducers)};
}
// Hashed support closure with breadth-first discovery.  For a fixed set of
// active leading monomials this computes exactly the same closure as layout(),
// but avoids ordered-map/set insertions during symbolic preprocessing.
// Each support monomial is enqueued once at minimum discovery depth.
static Layout indexed_symbolic_layout(Engine &e, const std::vector<Poly> &ss) {
  if (!active_mon_pool)
    throw std::runtime_error(
        "indexed symbolic preprocessing requires monomial interning");
  std::unordered_set<uint32_t> support;

  std::deque<std::pair<uint32_t, unsigned>> pending;
  size_t approximate = 0;
  for (const auto &f : ss)
    approximate += f.size();
  support.reserve(std::min(e.max_terms, approximate * 3 + 128));

  auto enqueue = [&](const Mon &m, unsigned depth) {
    const auto id = m.id();
    if (!id)
      throw std::runtime_error("missing interned symbolic monomial");
    bool inserted = false;
    inserted = support.insert(id).second;

    if (inserted)
      pending.emplace_back(id, depth);
  };
  for (const auto &f : ss)
    for (const auto &[mon, c] : f)
      enqueue(mon, 0);

  std::vector<Mon> leads;
  leads.reserve(e.basis.size());
  for (const auto &f : e.basis)
    leads.push_back(lead(f, e));
  std::vector<Reducer> reducers;
  while (!pending.empty()) {
    const auto [id, depth] = pending.front();
    pending.pop_front();

    const Mon mon = Mon::from_id(id);
    auto eligible = [&](size_t i) {
      const Mon &lm = leads[i];
      if (degree(lm) > degree(mon) || (lm.support_mask() & ~mon.support_mask()))
        return false;
      ++e.f4_reducer_candidates;
      if (!divides(lm, mon))
        return false;
      ++e.f4_reducer_divides;
      return true;
    };
    size_t chosen = SIZE_MAX;

    if (chosen == SIZE_MAX) {
      chosen = (ReducerSelection::first(DenseReducerCandidates{e.basis.size()},
                                        eligible));
    }
    if (chosen != SIZE_MAX) {
      const Mon shift = minus(mon, leads[chosen]);
      reducers.emplace_back(shift, chosen);

      {
        const auto &poly = e.basis[chosen];
        for (auto it = poly.begin(); it != poly.end(); ++it) {

          enqueue(plus(it->first, shift), depth + 1);
        }
      }
    }
    if (((support.size())) > e.max_terms || reducers.size() > e.max_terms)
      throw NativeBudgetError("native F4 symbolic budget exceeded");
  }
  std::vector<Mon> columns;
  {
    columns.reserve(support.size());
    for (uint32_t id : support)
      columns.push_back(Mon::from_id(id));
  }
  std::sort(columns.begin(), columns.end(), [&](const Mon &a, const Mon &b) {
    return compare(a, b, e.order) > 0;
  });
  std::sort(reducers.begin(), reducers.end(),
            [&](const Reducer &a, const Reducer &b) {
              return compare(plus(a.first, leads[a.second]),
                             plus(b.first, leads[b.second]), e.order) > 0;
            });

  ++e.symbolic_hash_batches;
  e.symbolic_hash_columns += columns.size();

  return {std::move(columns), std::move(reducers)};
}

// Direct CSR assembly from the shared, contiguous basis-term arena.
// Every matrix row is stored as a sorted vector of 8-byte (column, value)
// records. Pivots are appended to a second contiguous arena with stable
// offset/length spans. No tree nodes are allocated during elimination.
struct FlatColumn {
  uint32_t column;
  uint32_t coefficient;
};
static_assert(sizeof(FlatColumn) == 8, "matrix entries must be 8 bytes");
struct FlatSparseStorage {};

template <> struct SparseArithmetic<FlatSparseStorage> {
  static inline void subtract_shifted(std::vector<FlatColumn> &out,
                                      const std::vector<FlatColumn> &row,
                                      const std::vector<FlatColumn> &pivots,
                                      size_t start, size_t end, uint64_t scale,
                                      unsigned p) {
    constexpr uint32_t absent = UINT32_MAX;
    out.clear();
    size_t i = 0, j = start;
    while (i < row.size() || j < end) {
      const auto a = i < row.size() ? row[i].column : absent;
      const auto b = j < end ? pivots[j].column : absent;
      if (a < b) {
        out.push_back(row[i++]);
        continue;
      }
      if (b < a) {
        const uint32_t v =
            static_cast<uint32_t>((p - scale * pivots[j].coefficient % p) % p);
        if (v)
          out.push_back({b, v});
        ++j;
        continue;
      }
      const uint32_t v = static_cast<uint32_t>(
          (row[i].coefficient + p - scale * pivots[j].coefficient % p) % p);
      if (v)
        out.push_back({a, v});
      ++i;
      ++j;
    }
  }
};
static std::vector<Poly> reduce_contiguous_rows(const Layout &spec,
                                                const std::vector<Poly> &ss,
                                                Engine &e) {
  if ((!active_mon_pool))
    throw std::runtime_error(
        "flat backend requires contiguous basis and interned monomials");
  const auto width = spec.columns.size();
  if (width > static_cast<size_t>(UINT32_MAX))
    throw std::runtime_error("flat matrix column overflow");
  constexpr uint32_t absent = UINT32_MAX;
  std::vector<uint32_t> mon_to_column(active_mon_pool->by_id.size() + 1,
                                      absent);
  for (size_t col = 0; col < width; ++col) {
    const auto id = spec.columns[col].id();
    if (!id || id >= mon_to_column.size())
      throw std::runtime_error("flat matrix monomial not interned");
    mon_to_column[id] = static_cast<uint32_t>(col);
  }
  auto index_of = [&](const Mon &m) -> uint32_t {
    const auto id = m.id();
    if (!id || id >= mon_to_column.size() || mon_to_column[id] == absent)
      throw std::runtime_error("flat F4 symbolic layout misses a monomial");
    ++e.f4_index_lookup_hits;
    return mon_to_column[id];
  };
  const size_t nrows = spec.reducers.size() + ss.size();
  std::vector<size_t> row_offsets;
  row_offsets.reserve(nrows + 1);
  row_offsets.push_back(0);
  std::vector<FlatColumn> row_terms;
  std::vector<FlatColumn> raw;
  auto commit_row = [&]() {
    std::sort(raw.begin(), raw.end(),
              [](auto a, auto b) { return a.column < b.column; });
    for (size_t i = 1; i < raw.size(); ++i)
      if (raw[i].column == raw[i - 1].column)
        throw std::runtime_error("duplicate flat matrix column");
    row_terms.insert(row_terms.end(), raw.begin(), raw.end());
    row_offsets.push_back(row_terms.size());
    raw.clear();
  };
  for (const auto &[shift, i] : spec.reducers) {
    if (i + 1 >= e.flat_basis_offsets.size())
      throw std::runtime_error("flat basis index overflow");
    for (size_t j = e.flat_basis_offsets[i]; j < e.flat_basis_offsets[i + 1];
         ++j) {
      const auto &t = e.flat_basis_terms[j];
      raw.push_back(
          {index_of(plus(Mon::from_id(t.monomial), shift)), t.coefficient});
    }
    commit_row();
  }
  for (const auto &poly : ss) {
    for (const auto &[mon, c] : poly)
      raw.push_back({index_of(mon), static_cast<uint32_t>(c)});
    commit_row();
  }
  e.flat_matrix_terms += row_terms.size();
  std::vector<size_t> pivot_offset(width, SIZE_MAX);
  std::vector<size_t> pivot_length(width, 0);
  std::vector<FlatColumn> pivot_terms;

  constexpr uint32_t tile_width = 64;

  struct PivotTile {
    uint32_t first_column;
    uint32_t offset;
    uint16_t nnz;
    bool dense;
  };
  std::vector<PivotTile> tiles;
  std::vector<uint32_t> dense_tiles;
  std::vector<size_t> tile_start(width, SIZE_MAX), tile_count(width, 0);
  size_t pivot_count = 0;
  auto register_tiled_pivot = [&](uint32_t leading, size_t begin,
                                  size_t count) {
    ++pivot_count;

    tile_start[leading] = tiles.size();
    const size_t end = begin + count;
    for (size_t pos = begin; pos < end;) {
      const uint32_t first =
          (pivot_terms[pos].column / tile_width) * tile_width;
      size_t last = pos;
      while (last < end && pivot_terms[last].column < first + tile_width)
        ++last;
      const size_t length = last - pos;
      // A dense tile is beneficial only for high occupancy. Reducer rows
      // below the threshold continue using the zero-copy sparse CSR view.
      constexpr double required_occupancy = 0.625;
      const bool use_dense =
          length >=
              static_cast<size_t>(std::ceil(required_occupancy * tile_width)) &&
          first + tile_width <= width;
      if (use_dense) {
        if (dense_tiles.size() > UINT32_MAX - tile_width)
          throw NativeBudgetError("F4 tile cache budget exceeded");
        const uint32_t dense_offset = static_cast<uint32_t>(dense_tiles.size());
        dense_tiles.resize(dense_tiles.size() + tile_width, 0);
        if (dense_tiles.size() > e.max_terms * 32)
          throw NativeBudgetError("F4 dense pivot tile memory budget exceeded");
        for (size_t j = pos; j < last; ++j)
          dense_tiles[dense_offset + pivot_terms[j].column - first] =
              pivot_terms[j].coefficient;
        tiles.push_back(
            {first, dense_offset, static_cast<uint16_t>(length), true});
        ++e.matrix_tile_dense;
      } else {
        if (pos > UINT32_MAX)
          throw std::runtime_error("F4 sparse tile offset overflow");
        tiles.push_back({first, static_cast<uint32_t>(pos),
                         static_cast<uint16_t>(length), false});
        ++e.matrix_tile_sparse;
      }
      pos = last;
      if (tiles.size() > e.max_terms * 32)
        throw NativeBudgetError("F4 pivot tile metadata budget exceeded");
    }
    tile_count[leading] = tiles.size() - tile_start[leading];
  };
  std::vector<FlatColumn> row, temp;
  std::vector<uint32_t> dense_reused;
  std::vector<FlatColumn> committed_reused;

  std::vector<Poly> echelon;
  echelon.reserve(std::min(width, nrows));
  std::vector<std::vector<Mon>> paths;

  for (size_t row_idx = 0; row_idx < nrows; ++row_idx) {
    row.assign(row_terms.begin() + row_offsets[row_idx],
               row_terms.begin() + row_offsets[row_idx + 1]);
    const bool work_row = row_idx >= spec.reducers.size();
    if (work_row) {
      ++e.block_work_rows;
      e.block_work_nnz += row.size();
      e.block_work_cells += width;
      if (row.size() * 5 >= width)
        ++e.block_work_ge20pct;
      if (row.size() * 5 >= width * 2)
        ++e.block_work_ge40pct;
    } else {
      ++e.block_reducer_rows;
      e.block_reducer_nnz += row.size();
      e.block_reducer_cells += width;
    }
    const double row_density =
        width ? static_cast<double>(row.size()) / width : 0.0;
    const double pivot_density =
        (width && pivot_count)
            ? static_cast<double>(pivot_terms.size()) / (width * pivot_count)
            : 0.0;
    const double tiled_threshold =
        (std::clamp(0.20 - 0.20 * pivot_density + (width < 512 ? 0.04 : -0.025),
                    0.10, 0.26));
    const bool dense_work =
        (work_row) && width <= 8192 &&
        ((((width >= 192) && row_density >= tiled_threshold)));
    if (dense_work) {
      ++e.block_dense_rows;
      ++e.matrix_tiled_rows;
      std::vector<uint32_t> dense_local;
      std::vector<uint32_t> &dense = (dense_local);
      dense.resize(width, 0);
      for (const auto &t : row)
        dense[t.column] = t.coefficient;
      for (size_t column = 0; column < width; ++column) {
        const uint32_t coefficient = dense[column];
        if (!coefficient)
          continue;
        if (e.step_limit())
          throw NativeBudgetError(
              "F4 block dense reduction step budget exceeded");
        const size_t pivot_start = pivot_offset[column];
        if (pivot_start != SIZE_MAX) {
          ++e.flat_row_reductions;
          {
            ++e.matrix_tile_updates;
            if (tile_start[column] == SIZE_MAX)
              throw std::runtime_error("missing F4 pivot tile metadata");
            for (size_t k = tile_start[column];
                 k < tile_start[column] + tile_count[column]; ++k) {
              const auto &block = tiles[k];
              if (block.dense) {
                // 64 contiguous positions form the cache-local multiplication
                // tile.
                for (uint32_t j = 0; j < tile_width; ++j) {
                  const uint64_t amount = static_cast<uint64_t>(coefficient) *
                                          dense_tiles[block.offset + j] % e.p;
                  dense[block.first_column + j] = static_cast<uint32_t>(
                      (static_cast<uint64_t>(dense[block.first_column + j]) +
                       e.p - amount) %
                      e.p);
                }
              } else {
                for (size_t j = block.offset;
                     j < static_cast<size_t>(block.offset) + block.nnz; ++j) {
                  const auto &t = pivot_terms[j];
                  const uint32_t amount = static_cast<uint32_t>(
                      static_cast<uint64_t>(coefficient) * t.coefficient % e.p);
                  dense[t.column] = static_cast<uint32_t>(
                      (static_cast<uint64_t>(dense[t.column]) + e.p - amount) %
                      e.p);
                }
              }
            }
          }
          continue;
        }
        const uint64_t inv = inverse(coefficient, e.p);
        std::vector<FlatColumn> local_committed;
        std::vector<FlatColumn> &committed = (local_committed);
        committed.clear();
        ;
        for (size_t j = column; j < width; ++j) {
          if (dense[j])
            committed.push_back({static_cast<uint32_t>(j),
                                 static_cast<uint32_t>(dense[j] * inv % e.p)});
        }
        if (committed.size() > e.max_terms)
          throw NativeBudgetError("F4 block dense row term budget exceeded");
        pivot_offset[column] = pivot_terms.size();
        pivot_length[column] = committed.size();
        pivot_terms.insert(pivot_terms.end(), committed.begin(),
                           committed.end());
        register_tiled_pivot(static_cast<uint32_t>(column),
                             pivot_offset[column], pivot_length[column]);
        e.flat_pivot_terms_peak =
            std::max(e.flat_pivot_terms_peak, pivot_terms.size());
        if (pivot_terms.size() > e.max_terms * 32)
          throw NativeBudgetError(
              "F4 contiguous pivot-term memory budget exceeded");
        Poly polynomial;
        for (const auto &t : committed)
          polynomial.emplace(spec.columns[t.column], t.coefficient);
        echelon.push_back(std::move(polynomial));
        ++e.block_work_pivots;
        break;
      }
      continue;
    }
    ++e.block_sparse_rows;
    while (!row.empty()) {
      if (e.step_limit())
        throw NativeBudgetError(
            "F4 contiguous matrix reduction step budget exceeded");
      const uint32_t lead_col = row.front().column;

      const size_t pivot_start = pivot_offset[lead_col];
      if (pivot_start == SIZE_MAX) {
        const uint64_t inv = inverse(row.front().coefficient, e.p);
        pivot_offset[lead_col] = pivot_terms.size();
        pivot_length[lead_col] = row.size();
        Poly polynomial;
        for (auto &term : row) {
          term.coefficient =
              static_cast<uint32_t>(term.coefficient * inv % e.p);
          polynomial.emplace(spec.columns[term.column], term.coefficient);
        }
        pivot_terms.insert(pivot_terms.end(), row.begin(), row.end());
        register_tiled_pivot(lead_col, pivot_offset[lead_col],
                             pivot_length[lead_col]);
        e.flat_pivot_terms_peak =
            std::max(e.flat_pivot_terms_peak, pivot_terms.size());
        if (pivot_terms.size() > e.max_terms * 32)
          throw NativeBudgetError(
              "F4 contiguous pivot-term memory budget exceeded");
        // Only S-origin rows can contribute new information to this batch.
        // Reducer-origin rows remain as exact pivots, but are not candidates
        // for basis insertion. Every original reducer row is a basis multiple.
        if (row_idx >= spec.reducers.size()) {
          echelon.push_back(std::move(polynomial));
          ++e.block_work_pivots;
        } else {
          ++e.flat_reducer_pivots_excluded;
          ++e.block_reducer_pivots;
        }
        break;
      }
      ++e.flat_row_reductions;
      const uint64_t scale = row.front().coefficient;
      const size_t end = pivot_start + pivot_length[lead_col];
      SparseArithmetic<FlatSparseStorage>::subtract_shifted(
          temp, row, pivot_terms, pivot_start, end, scale, e.p);
      if (temp.size() > e.max_terms)
        throw NativeBudgetError("F4 contiguous row term budget exceeded");
      row.swap(temp);
    }
  }

  ++e.flat_matrix_batches;
  return echelon;
}

static void finish_basis(Engine &e);
static uint64_t f4_elapsed_ns(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}
static void run(Engine &e) {
  native_check_signals();
  while ((!e.f4_degree_queue.empty())) {
    native_check_signals();
    auto start = std::chrono::steady_clock::now();
    std::vector<Pair> take;
    {
      auto bucket = e.f4_degree_queue.begin();
      auto &entries = bucket->second;
      const size_t cap = (e.batch);
      while (!entries.empty() && take.size() < cap) {
        take.push_back(std::move(entries.front()));
        entries.pop_front();
        --e.f4_degree_queue_size;
      }
      if (entries.empty())
        e.f4_degree_queue.erase(bucket);
    }
    e.f4_pair_select_ns += f4_elapsed_ns(start);
    e.pairs += take.size();
    if (e.pairs > e.max_pairs)
      throw NativeBudgetError("native F4 pair limit");
    start = std::chrono::steady_clock::now();
    std::vector<Poly> ss;
    for (const auto &[i, j, m] : take) {
      Poly s = ((e.basis.size() > 64))
                   ? merged_s_poly(e.basis[i], e.basis[j], m, e)
                   : s_poly(e.basis[i], e.basis[j], m, e);
      if (!s.empty())
        ss.push_back(std::move(s));
    }
    e.f4_spoly_ns += f4_elapsed_ns(start);

    size_t lcm_min = SIZE_MAX, lcm_max = 0;
    if (e.collect_batch_metrics)
      for (const auto &pair : take) {
        lcm_min =
            std::min(lcm_min, static_cast<size_t>(degree(std::get<2>(pair))));
        lcm_max =
            std::max(lcm_max, static_cast<size_t>(degree(std::get<2>(pair))));
      }
    if (ss.empty()) {
      if (e.collect_batch_metrics)
        e.batch_metrics.push_back({take.size(), lcm_min, lcm_max, 0,
                                   e.basis.size(), e.basis.size(), 0, 0, 0, 0});
      continue;
    }
    start = std::chrono::steady_clock::now();

    Layout spec = ((((e.basis.size() <= 128)))) ? indexed_symbolic_layout(e, ss)
                                                : layout(e, ss);
    const size_t symbolic_ns = f4_elapsed_ns(start);
    e.f4_symbolic_ns += symbolic_ns;
    const size_t matrix_rows = spec.reducers.size() + ss.size();

    e.layout_rows += spec.reducers.size() + ss.size();
    e.layout_columns += spec.columns.size();
    ++e.matrix_preprocessed_batches;
    e.matrix_peak_width = std::max(e.matrix_peak_width, spec.columns.size());
    e.matrix_peak_rows =
        std::max(e.matrix_peak_rows, spec.reducers.size() + ss.size());
    e.matrix_peak_reducer_rows =
        std::max(e.matrix_peak_reducer_rows, spec.reducers.size());

    start = std::chrono::steady_clock::now();
    std::vector<Poly> rows;

    e.f4_assembly_ns += f4_elapsed_ns(start);
    start = std::chrono::steady_clock::now();
    std::vector<Poly> echelon;

    {
      echelon = reduce_contiguous_rows(spec, ss, e);
    }
    const size_t elimination_ns = f4_elapsed_ns(start);
    e.f4_elimination_ns += elimination_ns;
    e.emitted_rows += echelon.size();
    e.matrix_peak_echelon = std::max(e.matrix_peak_echelon, echelon.size());
    start = std::chrono::steady_clock::now();
    const size_t basis_before = e.basis.size();
    for (auto &row : echelon)
      insert(e, std::move(row));
    const size_t insertion_ns = f4_elapsed_ns(start);
    if (e.collect_batch_metrics) {
      size_t max_degree = 0;
      for (const auto &m : spec.columns)
        max_degree = std::max(max_degree, static_cast<size_t>(degree(m)));
      e.batch_metrics.push_back({take.size(), lcm_min, lcm_max, matrix_rows,
                                 spec.columns.size(), basis_before,
                                 e.basis.size(), symbolic_ns, elimination_ns,
                                 insertion_ns, max_degree});
    }

    e.f4_insertion_ns += insertion_ns;
    ++e.batches;
  }
  auto start = std::chrono::steady_clock::now();
  finish_basis(e);
  e.f4_finish_ns += f4_elapsed_ns(start);
}
// For a completed Groebner basis G, a subset containing one polynomial
// for every MINIMAL leading monomial is itself a Groebner basis of <G>:
// it is contained in <G> and generates LT(<G>).  This is valid only at the
// completed Buchberger stage (never as a replacement for pair processing).
// A cached leading-term table is safe across all final normal forms because
// their divisor set is immutable until the pass has completed.
static Poly finish_nf_cached(Poly f, const std::vector<Poly> &divisors,
                             const std::vector<Mon> &leads,
                             const std::vector<uint64_t> &inverses, Engine &e,
                             size_t skip) {
  Poly remainder;
  while (!f.empty()) {
    if (e.step_limit())
      throw NativeBudgetError(
          "F4 minimal final reduction step budget exceeded");
    const Mon m = f.begin()->first;
    const uint64_t c = f.begin()->second;
    size_t selected = divisors.size();
    for (size_t i = 0; i < divisors.size(); ++i) {
      if (i == skip || !inverses[i])
        continue;
      if (divides(leads[i], m)) {
        selected = i;
        break;
      }
    }
    if (selected == divisors.size()) {
      remainder.emplace(m, c);
      f.erase(f.begin());
    } else
      subtract_multiple(f, divisors[selected], minus(m, leads[selected]),
                        c * inverses[selected] % e.p, e);
    if (remainder.size() > e.max_terms)
      throw NativeBudgetError("F4 final remainder budget exceeded");
  }
  return remainder;
}
static void finish_basis_minimal(Engine &e) {
  e.pre_interreduction_basis = e.basis.size();
  const size_t n = e.basis.size();
  std::vector<unsigned char> keep(n, 1);
  {
    // When leading terms coincide, deterministically retain the earliest row.
    // Minimality is global, not a one-pass greedy deletion: a deleted term may
    // not justify deletion of another unless a surviving minimal term covers
    // it.
    for (size_t i = 0; i < n; ++i) {
      if (e.basis[i].empty()) {
        keep[i] = 0;
        continue;
      }
      const Mon &lm = e.basis[i].begin()->first;
      for (size_t j = 0; j < n; ++j) {
        if (i == j || e.basis[j].empty())
          continue;
        const Mon &other = e.basis[j].begin()->first;
        if (divides(other, lm) && (other != lm || j < i)) {
          keep[i] = 0;
          break;
        }
      }
    }
  }
  std::vector<Poly> basis;
  basis.reserve(n);
  for (size_t i = 0; i < n; ++i)
    if (keep[i])
      basis.push_back(std::move(e.basis[i]));
  e.finish_pruned_lm = n - basis.size();
  std::vector<Mon> leads;
  std::vector<uint64_t> inverses;
  leads.reserve(basis.size());
  inverses.reserve(basis.size());
  for (const auto &f : basis) {
    leads.push_back(f.begin()->first);
    inverses.push_back(inverse(f.begin()->second, e.p));
  }
  std::vector<Poly> reduced;
  reduced.reserve(basis.size());
  for (size_t i = 0; i < basis.size(); ++i) {
    // A polynomial whose tail is already reduced needs only normalization.
    bool tail_reduced = true;
    auto term = std::next(basis[i].begin());
    for (; term != basis[i].end() && tail_reduced; ++term) {
      for (size_t j = 0; j < basis.size(); ++j) {
        if (j != i && divides(leads[j], term->first)) {
          tail_reduced = false;
          break;
        }
      }
    }
    Poly f;
    if (tail_reduced) {
      f = basis[i];
      ++e.finish_skipped_nf;
    } else {
      ++e.finish_nf_calls;
      f = finish_nf_cached(basis[i], basis, leads, inverses, e, i);
    }
    if (f.empty() || f.begin()->first != leads[i])
      throw std::runtime_error(
          "F4 minimal final basis lost a leading monomial");
    monic(f, e);
    reduced.push_back(std::move(f));
  }
  std::sort(reduced.begin(), reduced.end(), [&](const auto &a, const auto &b) {
    return compare(a.begin()->first, b.begin()->first, e.order) > 0;
  });
  // These cheap postconditions are independent of the reduction path.
  for (size_t i = 0; i < reduced.size(); ++i) {
    for (auto term = std::next(reduced[i].begin()); term != reduced[i].end();
         ++term)
      for (size_t j = 0; j < reduced.size(); ++j)
        if (divides(reduced[j].begin()->first, term->first))
          throw std::runtime_error("F4 minimal final basis has reducible tail");
  }
  e.basis = std::move(reduced);
  ++e.f4_reduction_passes;
}
static void finish_basis(Engine &e) {
  {
    finish_basis_minimal(e);
    return;
  }
}

#include "_native_f4_bindings.inc"
