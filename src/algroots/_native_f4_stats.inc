// Statistics for the single native F4 kernel.
setstat("pairs_processed", e.pairs);
setstat("batches", e.batches);
setstat("max_queued_pairs", e.max_queued_pairs);
setstat("normal_form_steps", e.nf_steps);
setstat("chain_pairs_pruned", e.chain_pairs_pruned);
setstat("prospective_chain_pairs", e.prospective_chain_pairs);
setstat("pair_index_candidates", e.pair_index_candidates);
setstat("matrix_peak_width", e.matrix_peak_width);
setstat("matrix_peak_rows", e.matrix_peak_rows);
setstat("matrix_peak_reducer_rows", e.matrix_peak_reducer_rows);
setstat("matrix_peak_echelon", e.matrix_peak_echelon);
setstat("peak_intermediate_basis", e.peak_intermediate_basis);
setstat("pre_interreduction_basis", e.pre_interreduction_basis);
setstat("finish_pruned_lm", e.finish_pruned_lm);
setstat("finish_skipped_nf", e.finish_skipped_nf);
setstat("finish_nf_calls", e.finish_nf_calls);
setstat("symbolic_rows", e.layout_rows);
setstat("symbolic_columns", e.layout_columns);
setstat("emitted_echelon_rows", e.emitted_rows);
setstat("f4_pair_selection_ns", e.f4_pair_select_ns);
setstat("f4_s_polynomial_ns", e.f4_spoly_ns);
setstat("f4_symbolic_preprocessing_ns", e.f4_symbolic_ns);
setstat("f4_row_assembly_ns", e.f4_assembly_ns);
setstat("f4_matrix_elimination_ns", e.f4_elimination_ns);
setstat("f4_basis_insertion_ns", e.f4_insertion_ns);
setstat("f4_final_interreduction_ns", e.f4_finish_ns);
setstat("matrix_tile_sparse", e.matrix_tile_sparse);
setstat("matrix_tile_dense", e.matrix_tile_dense);
setstat("matrix_tile_updates", e.matrix_tile_updates);
setstat("block_sparse_rows", e.block_sparse_rows);
setstat("block_dense_rows", e.block_dense_rows);
setstat("interned_monomials", pool.by_id.size());
setstat("monomial_cache_hits", pool.hits);
setstat("monomial_cache_misses", pool.misses);
auto metrics = own_python(PyList_New(0));
for (const auto &values : e.batch_metrics) {
  auto row = own_python(PyTuple_New(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    auto value = own_python(PyLong_FromSize_t(values[i]));
    PyTuple_SET_ITEM(row.get(), i, value.release());
  }
  check_python(PyList_Append(metrics.get(), row.get()));
}
check_python(PyDict_SetItemString(stats, "batch_metrics", metrics.get()));
