// Do not include torch/extension.h or pybind11 headers here.
// vLLM builds _C with Py_LIMITED_API; pulling pybind11 into a CUDA TU
// causes Py_buffer / PyObject_* symbols to be unavailable.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

template <typename T>
__device__ inline float to_float(T v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float v) {
  return static_cast<scalar_t>(v);
}

__device__ inline int find_request_for_query_token(
    int q_token,
    int q_tokens,
    int active,
    const int32_t* __restrict__ req_token_lens) {
  if (active <= 0) {
    return -1;
  }
  // Common decode path: one query token per active request.
  if (q_tokens == active) {
    return q_token;
  }
  int acc = 0;
  for (int r = 0; r < active; ++r) {
    int len = req_token_lens[r];
    if (q_token < acc + len) {
      return r;
    }
    acc += len;
  }
  return active - 1;
}

template <typename scalar_t>
__global__ void sparse_ssd_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ active_reqs,
    const int32_t* __restrict__ req_token_lens,
    const int32_t* __restrict__ selected_block_table,
    const int32_t* __restrict__ selected_block_lens,
    const int32_t* __restrict__ selected_ready_flags,
    int64_t q_s0,
    int64_t q_s1,
    int64_t q_s2,
    int64_t out_s0,
    int64_t out_s1,
    int64_t out_s2,
    int64_t kv_s0,
    int64_t kv_s1,
    int64_t kv_s2,
    int64_t kv_s3,
    int64_t kv_s4,
    int kv_layout,
    int q_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_selected_blocks,
    float scale) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = q_tokens * num_q_heads;
  if (linear >= total) {
    return;
  }

  const int q_token = linear / num_q_heads;
  const int q_head = linear - q_token * num_q_heads;
  const int active = active_reqs[0];

  const int req = find_request_for_query_token(q_token, q_tokens, active, req_token_lens);
  const bool valid_req = (req >= 0 && req < active);
  const int ready = valid_req ? selected_ready_flags[req] : 0;
  const int selected_len = valid_req ? selected_block_lens[req] : 0;

  if (!valid_req || ready == 0 || selected_len <= 0) {
    for (int d = 0; d < head_dim; ++d) {
      out[q_token * out_s0 + q_head * out_s1 + d * out_s2] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int kv_head = (num_kv_heads == num_q_heads)
      ? q_head
      : (q_head % (num_kv_heads > 0 ? num_kv_heads : 1));

  float max_score = -FLT_MAX;
  for (int bi = 0; bi < selected_len && bi < max_selected_blocks; ++bi) {
    const int block_id = selected_block_table[req * max_selected_blocks + bi];
    if (block_id < 0) {
      continue;
    }
    for (int t = 0; t < block_size; ++t) {
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const float qv = to_float(query[q_token * q_s0 + q_head * q_s1 + d * q_s2]);
        int64_t k_off;
        if (kv_layout == 0) {  // [2, num_blocks, block, kv_heads, head]
          k_off = 0 * kv_s0 + static_cast<int64_t>(block_id) * kv_s1 +
                  t * kv_s2 + kv_head * kv_s3 + d * kv_s4;
        } else {  // [num_blocks, 2, block, kv_heads, head]
          k_off = static_cast<int64_t>(block_id) * kv_s0 + 0 * kv_s1 +
                  t * kv_s2 + kv_head * kv_s3 + d * kv_s4;
        }
        score += qv * to_float(kv_cache[k_off]);
      }
      score *= scale;
      max_score = fmaxf(max_score, score);
    }
  }

  float denom = 0.0f;
  // Accumulate each output dimension independently to keep the implementation
  // simple and graph-safe. This is a correctness/reference CUDA kernel; replace
  // with a tiled optimized kernel after validating metadata and selected-load.
  for (int d = 0; d < head_dim; ++d) {
    float acc = 0.0f;
    denom = 0.0f;
    for (int bi = 0; bi < selected_len && bi < max_selected_blocks; ++bi) {
      const int block_id = selected_block_table[req * max_selected_blocks + bi];
      if (block_id < 0) {
        continue;
      }
      for (int t = 0; t < block_size; ++t) {
        float score = 0.0f;
        for (int kd = 0; kd < head_dim; ++kd) {
          const float qv = to_float(query[q_token * q_s0 + q_head * q_s1 + kd * q_s2]);
          int64_t k_off;
          if (kv_layout == 0) {
            k_off = 0 * kv_s0 + static_cast<int64_t>(block_id) * kv_s1 +
                    t * kv_s2 + kv_head * kv_s3 + kd * kv_s4;
          } else {
            k_off = static_cast<int64_t>(block_id) * kv_s0 + 0 * kv_s1 +
                    t * kv_s2 + kv_head * kv_s3 + kd * kv_s4;
          }
          score += qv * to_float(kv_cache[k_off]);
        }
        score *= scale;
        const float w = expf(score - max_score);
        denom += w;
        int64_t v_off;
        if (kv_layout == 0) {
          v_off = 1 * kv_s0 + static_cast<int64_t>(block_id) * kv_s1 +
                  t * kv_s2 + kv_head * kv_s3 + d * kv_s4;
        } else {
          v_off = static_cast<int64_t>(block_id) * kv_s0 + 1 * kv_s1 +
                  t * kv_s2 + kv_head * kv_s3 + d * kv_s4;
        }
        acc += w * to_float(kv_cache[v_off]);
      }
    }
    const float out_v = (denom > 0.0f) ? (acc / denom) : 0.0f;
    out[q_token * out_s0 + q_head * out_s1 + d * out_s2] = from_float<scalar_t>(out_v);
  }
}

}  // namespace

void sparse_ssd_attention(
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
    int64_t block_size,
    int64_t chunk_size,
    int64_t top_n_chunks,
    double scale) {
  (void)req_vllm_cached_tokens;
  (void)req_lmcache_cached_tokens;
  (void)req_slot_lens;
  (void)slot_mapping_table;
  (void)chunk_size;
  (void)top_n_chunks;

  TORCH_CHECK(out.is_cuda(), "sparse_ssd_attention: out must be CUDA");
  TORCH_CHECK(query.is_cuda(), "sparse_ssd_attention: query must be CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "sparse_ssd_attention: kv_cache must be CUDA");
  TORCH_CHECK(active_reqs.is_cuda(), "sparse_ssd_attention: active_reqs must be CUDA");
  TORCH_CHECK(selected_block_table.is_cuda(), "sparse_ssd_attention: selected_block_table must be CUDA");
  TORCH_CHECK(selected_block_lens.is_cuda(), "sparse_ssd_attention: selected_block_lens must be CUDA");
  TORCH_CHECK(selected_ready_flags.is_cuda(), "sparse_ssd_attention: selected_ready_flags must be CUDA");
  TORCH_CHECK(query.dim() == 3, "sparse_ssd_attention: query must be [tokens, heads, head_dim]");
  TORCH_CHECK(out.dim() == 3, "sparse_ssd_attention: out must be [tokens, heads, head_dim]");
  TORCH_CHECK(kv_cache.dim() == 5, "sparse_ssd_attention: kv_cache must be 5-D");
  TORCH_CHECK(selected_block_table.dim() == 2, "sparse_ssd_attention: selected_block_table must be [reqs, blocks]");

  const c10::cuda::CUDAGuard device_guard(query.device());

  const int q_tokens = static_cast<int>(query.size(0));
  const int num_q_heads = static_cast<int>(query.size(1));
  const int head_dim = static_cast<int>(query.size(2));
  const int max_selected_blocks = static_cast<int>(selected_block_table.size(1));

  int kv_layout = -1;
  int num_kv_heads = 0;
  if (kv_cache.size(0) == 2) {
    kv_layout = 0;
    TORCH_CHECK(kv_cache.size(2) >= block_size, "kv_cache block dimension smaller than block_size");
    num_kv_heads = static_cast<int>(kv_cache.size(3));
  } else if (kv_cache.size(1) == 2) {
    kv_layout = 1;
    TORCH_CHECK(kv_cache.size(2) >= block_size, "kv_cache block dimension smaller than block_size");
    num_kv_heads = static_cast<int>(kv_cache.size(3));
  } else {
    TORCH_CHECK(false, "sparse_ssd_attention: unsupported kv_cache layout; expected [2,B,block,H,D] or [B,2,block,H,D]");
  }
  TORCH_CHECK(head_dim == kv_cache.size(4), "query head_dim must match kv_cache head_dim");

  const int total = q_tokens * num_q_heads;
  if (total == 0) {
    return;
  }
  const int threads = 128;
  const int blocks = (total + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_ssd_attention_cuda",
      [&] {
        sparse_ssd_attention_kernel<scalar_t><<<
            blocks,
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            kv_cache.data_ptr<scalar_t>(),
            active_reqs.data_ptr<int32_t>(),
            req_token_lens.data_ptr<int32_t>(),
            selected_block_table.data_ptr<int32_t>(),
            selected_block_lens.data_ptr<int32_t>(),
            selected_ready_flags.data_ptr<int32_t>(),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            kv_cache.stride(4),
            kv_layout,
            q_tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            static_cast<int>(block_size),
            max_selected_blocks,
            static_cast<float>(scale));
      });
}

// The operator schema is declared in csrc/torch_bindings.cpp together with
// vLLM's other _C schemas.  Keep only the CUDA implementation registration in
// this translation unit to avoid interfering with the main _C namespace
// registration.
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sparse_ssd_attention", &sparse_ssd_attention);
}
