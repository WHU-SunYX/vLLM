#pragma once

#include <torch/extension.h>

// Graph-capturable sparse SSD paged attention entry point.
// Host-side LMCache/GDS prepares selected KV blocks and stable device metadata
// before the model graph runs. This op only consumes those device tensors and
// never performs host I/O.
void sparse_ssd_attention(
    torch::Tensor& out,
    const torch::Tensor& query,
    const torch::Tensor& kv_cache,
    const torch::Tensor& active_reqs,
    const torch::Tensor& req_token_lens,
    const torch::Tensor& req_vllm_cached_tokens,
    const torch::Tensor& req_lmcache_cached_tokens,
    const torch::Tensor& req_slot_lens,
    const torch::Tensor& slot_mapping_table,
    const torch::Tensor& selected_block_table,
    const torch::Tensor& selected_block_lens,
    const torch::Tensor& selected_ready_flags,
    int64_t block_size,
    int64_t chunk_size,
    int64_t top_n_chunks,
    double scale);
