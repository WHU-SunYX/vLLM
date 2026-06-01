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
