#pragma once

#include <ATen/ATen.h>

// Standalone FlashAttention-style sparse selected-block paged attention.
// It consumes graph-stable device tensors prepared by LMCache/vLLM before
// CUDA graph replay and performs no host-side IO or Python callbacks.
void sparse_flash_attention(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& kv_cache,
    const at::Tensor& active_reqs,
    const at::Tensor& req_token_lens,
    const at::Tensor& req_vllm_cached_tokens,
    const at::Tensor& req_lmcache_cached_tokens,
    const at::Tensor& req_slot_lens,
    const at::Tensor& slot_mapping_table,
    const at::Tensor& selected_block_table,
    const at::Tensor& selected_block_lens,
    const at::Tensor& selected_ready_flags,
    const at::Tensor& debug_counters,
    int64_t block_size,
    int64_t chunk_size,
    int64_t top_n_chunks,
    double scale);


// Graph-side/HOST bridge for AI-SSD q-aware sparse KV selection.
// This reuses the existing sparse_flash_attention custom-op build unit instead
// of adding a second attention custom-op source. It must execute outside CUDA
// graph capture/replay because it performs D2H copy and HOST<->SSD CMB/RPC.
void aissd_sparse_kv_select(
    const at::Tensor& query,
    const at::Tensor& active_reqs,
    const at::Tensor& req_token_lens,
    const at::Tensor& req_lmcache_cached_tokens,
    const at::Tensor& aissd_candidate_count,
    const at::Tensor& aissd_candidate_chunk_ids,
    const at::Tensor& aissd_candidate_block_ids,
    const at::Tensor& aissd_candidate_block_lens,
    const at::Tensor& aissd_candidate_token_start,
    const at::Tensor& aissd_candidate_token_end,
    const at::Tensor& aissd_candidate_dtype,
    const at::Tensor& aissd_candidate_fmt,
    const at::Tensor& aissd_candidate_ndim,
    const at::Tensor& aissd_candidate_shape,
    const at::Tensor& aissd_candidate_extent_count,
    const at::Tensor& aissd_candidate_extent_lba,
    const at::Tensor& aissd_candidate_extent_bytes,
    at::Tensor& selected_block_table,
    at::Tensor& selected_block_lens,
    at::Tensor& selected_ready_flags,
    at::Tensor& fa_block_table,
    at::Tensor& fa_seq_lens,
    at::Tensor& aissd_rpc_selected_chunk_ids,
    at::Tensor& aissd_rpc_selected_chunk_lens,
    int64_t layer_id,
    int64_t backend,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t block_size,
    int64_t top_n_chunks,
    int64_t top_m,
    int64_t score_mode,
    int64_t manifest_block_size,
    int64_t timeout_ms);
