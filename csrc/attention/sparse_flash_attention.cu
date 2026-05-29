// Do not include torch/extension.h or pybind11 headers here.
// vLLM builds _C with Py_LIMITED_API; pulling pybind11 into a CUDA TU
// causes Py_buffer / PyObject_* symbols to be unavailable.
#include "attention/sparse_flash_attention.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kSparseFlashThreads = 128;

// Current first optimized version targets Qwen-style head_dim <= 128.
// Keeping this fixed lets each CUDA block map one thread to one head dimension
// and maintain one online-softmax accumulator per output dimension.
static_assert(kSparseFlashThreads == 128);

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
    const int len = max(req_token_lens[r], 0);
    if (q_token < acc + len) {
      return r;
    }
    acc += len;
  }
  // If req_token_lens describe cached/prompt lengths rather than current query
  // lengths, fall back to the last active request instead of reading OOB.
  return active - 1;
}

__device__ inline int map_q_head_to_kv_head(int q_head,
                                            int num_q_heads,
                                            int num_kv_heads) {
  if (num_kv_heads <= 0) {
    return 0;
  }
  if (num_kv_heads == num_q_heads) {
    return q_head;
  }
  // GQA/MQA convention: consecutive groups of query heads share one KV head.
  const int group = max(num_q_heads / num_kv_heads, 1);
  return min(q_head / group, num_kv_heads - 1);
}

template <typename scalar_t>
__device__ inline int64_t kv_offset(
    int kv_layout,
    int kv_kind,  // 0=K, 1=V
    int block_id,
    int token_in_block,
    int kv_head,
    int dim,
    int64_t kv_s0,
    int64_t kv_s1,
    int64_t kv_s2,
    int64_t kv_s3,
    int64_t kv_s4) {
  if (kv_layout == 0) {  // [2, num_blocks, block, kv_heads, head]
    return static_cast<int64_t>(kv_kind) * kv_s0 +
           static_cast<int64_t>(block_id) * kv_s1 +
           static_cast<int64_t>(token_in_block) * kv_s2 +
           static_cast<int64_t>(kv_head) * kv_s3 +
           static_cast<int64_t>(dim) * kv_s4;
  }
  // [num_blocks, 2, block, kv_heads, head]
  return static_cast<int64_t>(block_id) * kv_s0 +
         static_cast<int64_t>(kv_kind) * kv_s1 +
         static_cast<int64_t>(token_in_block) * kv_s2 +
         static_cast<int64_t>(kv_head) * kv_s3 +
         static_cast<int64_t>(dim) * kv_s4;
}

template <typename scalar_t>
__global__ void sparse_flash_attention_kernel(
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
  const int q_head = blockIdx.x;
  const int q_token = blockIdx.y;
  const int tid = threadIdx.x;

  if (q_token >= q_tokens || q_head >= num_q_heads) {
    return;
  }

  const int active = active_reqs[0];
  const int req = find_request_for_query_token(q_token, q_tokens, active,
                                               req_token_lens);
  const bool valid_req = (req >= 0 && req < active);
  const int ready = valid_req ? selected_ready_flags[req] : 0;
  const int selected_len = valid_req ? selected_block_lens[req] : 0;

  if (!valid_req || ready == 0 || selected_len <= 0) {
    for (int d = tid; d < head_dim; d += blockDim.x) {
      out[static_cast<int64_t>(q_token) * out_s0 +
          static_cast<int64_t>(q_head) * out_s1 +
          static_cast<int64_t>(d) * out_s2] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int kv_head = map_q_head_to_kv_head(q_head, num_q_heads, num_kv_heads);

  __shared__ float red[kSparseFlashThreads];
  __shared__ float score_s;

  // One thread owns one or more output dimensions. For the target path
  // head_dim=128 and blockDim=128, this is one accumulator per thread.
  float q_val = 0.0f;
  float acc = 0.0f;
  if (tid < head_dim) {
    q_val = to_float(query[static_cast<int64_t>(q_token) * q_s0 +
                           static_cast<int64_t>(q_head) * q_s1 +
                           static_cast<int64_t>(tid) * q_s2]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  const int bounded_selected_len = min(selected_len, max_selected_blocks);

  for (int bi = 0; bi < bounded_selected_len; ++bi) {
    const int block_id = selected_block_table[req * max_selected_blocks + bi];
    if (block_id < 0) {
      continue;
    }

    for (int token_in_block = 0; token_in_block < block_size; ++token_in_block) {
      float partial = 0.0f;
      if (tid < head_dim) {
        const int64_t k_off = kv_offset<scalar_t>(
            kv_layout, 0, block_id, token_in_block, kv_head, tid,
            kv_s0, kv_s1, kv_s2, kv_s3, kv_s4);
        partial = q_val * to_float(kv_cache[k_off]);
      }

      red[tid] = partial;
      __syncthreads();
      // Block-wide reduction to get QK for this selected KV token.
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
          red[tid] += red[tid + stride];
        }
        __syncthreads();
      }

      if (tid == 0) {
        score_s = red[0] * scale;
      }
      __syncthreads();
      const float score = score_s;

      // Online softmax update, FlashAttention-style. Each lane/thread keeps
      // the accumulator for its output dimension, while the scalar m_i/l_i are
      // identical across the block.
      const float m_new = fmaxf(m_i, score);
      const float alpha = (m_i == -FLT_MAX) ? 0.0f : expf(m_i - m_new);
      const float beta = expf(score - m_new);

      if (tid < head_dim) {
        const int64_t v_off = kv_offset<scalar_t>(
            kv_layout, 1, block_id, token_in_block, kv_head, tid,
            kv_s0, kv_s1, kv_s2, kv_s3, kv_s4);
        acc = acc * alpha + beta * to_float(kv_cache[v_off]);
      }
      l_i = l_i * alpha + beta;
      m_i = m_new;
      __syncthreads();
    }
  }

  if (tid < head_dim) {
    const float out_v = (l_i > 0.0f) ? (acc / l_i) : 0.0f;
    out[static_cast<int64_t>(q_token) * out_s0 +
        static_cast<int64_t>(q_head) * out_s1 +
        static_cast<int64_t>(tid) * out_s2] = from_float<scalar_t>(out_v);
  }
}

}  // namespace

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

  TORCH_CHECK(out.is_cuda(), "sparse_flash_attention: out must be CUDA");
  TORCH_CHECK(query.is_cuda(), "sparse_flash_attention: query must be CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "sparse_flash_attention: kv_cache must be CUDA");
  TORCH_CHECK(active_reqs.is_cuda(), "sparse_flash_attention: active_reqs must be CUDA");
  TORCH_CHECK(req_token_lens.is_cuda(), "sparse_flash_attention: req_token_lens must be CUDA");
  TORCH_CHECK(selected_block_table.is_cuda(), "sparse_flash_attention: selected_block_table must be CUDA");
  TORCH_CHECK(selected_block_lens.is_cuda(), "sparse_flash_attention: selected_block_lens must be CUDA");
  TORCH_CHECK(selected_ready_flags.is_cuda(), "sparse_flash_attention: selected_ready_flags must be CUDA");
  TORCH_CHECK(query.dim() == 3, "sparse_flash_attention: query must be [tokens, heads, head_dim]");
  TORCH_CHECK(out.dim() == 3, "sparse_flash_attention: out must be [tokens, heads, head_dim]");
  TORCH_CHECK(kv_cache.dim() == 5, "sparse_flash_attention: kv_cache must be 5-D");
  TORCH_CHECK(selected_block_table.dim() == 2, "sparse_flash_attention: selected_block_table must be [reqs, blocks]");
  TORCH_CHECK(selected_block_lens.dim() == 1, "sparse_flash_attention: selected_block_lens must be [reqs]");
  TORCH_CHECK(selected_ready_flags.dim() == 1, "sparse_flash_attention: selected_ready_flags must be [reqs]");

  const c10::cuda::CUDAGuard device_guard(query.device());

  const int q_tokens = static_cast<int>(query.size(0));
  const int num_q_heads = static_cast<int>(query.size(1));
  const int head_dim = static_cast<int>(query.size(2));
  const int max_selected_blocks = static_cast<int>(selected_block_table.size(1));

  TORCH_CHECK(out.size(0) == q_tokens && out.size(1) == num_q_heads &&
                  out.size(2) == head_dim,
              "sparse_flash_attention: out shape must match query shape");
  TORCH_CHECK(head_dim > 0 && head_dim <= kSparseFlashThreads,
              "sparse_flash_attention: first optimized kernel supports head_dim <= 128");
  TORCH_CHECK(block_size > 0, "sparse_flash_attention: block_size must be positive");

  int kv_layout = -1;
  int num_kv_heads = 0;
  if (kv_cache.size(0) == 2) {
    kv_layout = 0;
    TORCH_CHECK(kv_cache.size(2) >= block_size,
                "kv_cache block dimension smaller than block_size");
    num_kv_heads = static_cast<int>(kv_cache.size(3));
    TORCH_CHECK(head_dim == kv_cache.size(4),
                "query head_dim must match kv_cache head_dim");
  } else if (kv_cache.size(1) == 2) {
    kv_layout = 1;
    TORCH_CHECK(kv_cache.size(2) >= block_size,
                "kv_cache block dimension smaller than block_size");
    num_kv_heads = static_cast<int>(kv_cache.size(3));
    TORCH_CHECK(head_dim == kv_cache.size(4),
                "query head_dim must match kv_cache head_dim");
  } else {
    TORCH_CHECK(false,
                "sparse_flash_attention: unsupported kv_cache layout; expected "
                "[2,B,block,H,D] or [B,2,block,H,D]");
  }

  if (q_tokens == 0 || num_q_heads == 0) {
    return;
  }

  const dim3 grid(num_q_heads, q_tokens, 1);
  const dim3 block(kSparseFlashThreads, 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_flash_attention_cuda",
      [&] {
        sparse_flash_attention_kernel<scalar_t><<<
            grid,
            block,
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


TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sparse_flash_attention", &sparse_flash_attention);
}
