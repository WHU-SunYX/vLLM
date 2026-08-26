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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSparseFlashThreads = 128;

// Current first optimized version targets Qwen-style head_dim <= 128.
// Keeping this fixed lets each CUDA block map one thread to one head dimension
// and maintain one online-softmax accumulator per output dimension.
static_assert(kSparseFlashThreads == 128);

__device__ inline void debug_atomic_add(int64_t* counters, int idx, unsigned long long val) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&counters[idx]), val);
}

__device__ inline void debug_atomic_max(int64_t* counters, int idx, unsigned long long val) {
  atomicMax(reinterpret_cast<unsigned long long*>(&counters[idx]), val);
}

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
    int64_t* __restrict__ debug_counters,
    int debug_enabled,
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

  if (debug_enabled && tid == 0) {
    // debug_counters layout, accumulated per op call after Python zero_():
    // [0] launched query-head blocks
    // [1] active query-head blocks that will visit selected KV blocks
    // [2] selected block visits, accumulated across query-head blocks
    // [3] selected KV token visits, accumulated across query-head blocks
    // [4] inactive/empty query-head blocks
    // [5] invalid selected block entries encountered
    // [6] max active_reqs value observed inside the CUDA kernel
    // [7] max selected_block_lens value observed inside the CUDA kernel
    debug_atomic_add(debug_counters, 0, 1ULL);
    debug_atomic_max(debug_counters, 6, static_cast<unsigned long long>(max(active, 0)));
  }

  if (!valid_req || ready == 0 || selected_len <= 0) {
    if (debug_enabled && tid == 0) {
      debug_atomic_add(debug_counters, 4, 1ULL);
    }
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

  if (debug_enabled && tid == 0) {
    debug_atomic_add(debug_counters, 1, 1ULL);
    debug_atomic_add(debug_counters, 2, static_cast<unsigned long long>(bounded_selected_len));
    debug_atomic_add(debug_counters, 3,
                     static_cast<unsigned long long>(bounded_selected_len) *
                         static_cast<unsigned long long>(block_size));
    debug_atomic_max(debug_counters, 7,
                     static_cast<unsigned long long>(max(bounded_selected_len, 0)));
  }

  for (int bi = 0; bi < bounded_selected_len; ++bi) {
    const int block_id = selected_block_table[req * max_selected_blocks + bi];
    if (block_id < 0) {
      if (debug_enabled && tid == 0) {
        debug_atomic_add(debug_counters, 5, 1ULL);
      }
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
    const at::Tensor& debug_counters,
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
  const bool debug_enabled = debug_counters.defined() && debug_counters.numel() >= 8;
  if (debug_enabled) {
    TORCH_CHECK(debug_counters.is_cuda(), "sparse_flash_attention: debug_counters must be CUDA when provided");
    TORCH_CHECK(debug_counters.scalar_type() == at::ScalarType::Long,
                "sparse_flash_attention: debug_counters must be int64/torch.long");
    TORCH_CHECK(debug_counters.is_contiguous(),
                "sparse_flash_attention: debug_counters must be contiguous");
  }
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
            debug_enabled ? debug_counters.data_ptr<int64_t>() : nullptr,
            debug_enabled ? 1 : 0,
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


namespace {

constexpr uint32_t AISSD_SPARSE_KV_PROTOCOL_VERSION = 2u;
constexpr uint32_t AISSD_SPARSE_KV_FLAG_Q_INLINE_CMB = (1u << 0);
constexpr uint32_t AISSD_SPARSE_KV_FLAG_NATIVE_EXTENTS = (1u << 1);
constexpr uint32_t AISSD_SPARSE_KV_FLAG_RESULT_INLINE_CMB = (1u << 2);
constexpr int32_t CMD_SPARSE_KV_RUN_MANIFEST_LBA = 100;
constexpr uint32_t AISSD_MAX_DIMS = 8;
constexpr uint32_t AISSD_MAX_CANDIDATES = 256;
constexpr uint32_t AISSD_MAX_EXTENTS_PER_CANDIDATE = 64;
constexpr uint32_t AISSD_MAX_SELECTED_CHUNKS = 256;
constexpr uint32_t AISSD_MAX_SELECTED_BLOCKS = 4096;

#pragma pack(push, 1)
struct AissdSparseKvExtent {
  uint64_t lba;
  uint64_t bytes;
};

struct AissdSparseKvCandidate {
  uint32_t chunk_index;
  uint32_t token_start;
  uint32_t token_end;
  uint32_t num_tokens;
  uint32_t dtype;
  uint32_t fmt;
  uint32_t ndim;
  uint32_t shape[AISSD_MAX_DIMS];
  uint64_t file_offset;
  uint64_t nbytes;
  uint32_t extent_count;
  uint32_t reserved0;
  AissdSparseKvExtent extents[AISSD_MAX_EXTENTS_PER_CANDIDATE];
};

struct AissdSparseKvRunReq {
  int32_t cmd;
  uint32_t version;
  uint64_t job_id;
  uint64_t request_id;
  uint32_t layer_id;
  uint32_t backend;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t chunk_size;
  uint32_t block_size;
  uint32_t top_n_chunks;
  uint32_t top_m;
  uint32_t score_mode;
  uint32_t q_dtype;
  uint32_t kv_dtype;
  uint32_t q_token_count;
  uint32_t candidate_chunk_count;
  uint64_t q_cmb_offset;
  uint32_t q_bytes;
  uint32_t manifest_block_size;
  uint32_t flags;
  uint32_t reserved1;
  AissdSparseKvCandidate candidates[AISSD_MAX_CANDIDATES];
};

struct AissdSparseKvRunResp {
  int32_t status;
  uint32_t version;
  uint64_t job_id;
  uint64_t request_id;
  uint32_t layer_id;
  uint32_t backend;
  uint32_t selected_chunk_count;
  uint32_t selected_block_count;
  uint32_t block_size;
  uint32_t chunk_size;
  uint32_t error_code;
  uint32_t reserved0;
  uint32_t selected_chunk_ids[AISSD_MAX_SELECTED_CHUNKS];
  float selected_chunk_scores[AISSD_MAX_SELECTED_CHUNKS];
  uint32_t selected_block_ids[AISSD_MAX_SELECTED_BLOCKS];
};
#pragma pack(pop)

using AissdRunNativeExtentsFn = int (*)(const void*, uint32_t,
                                        const AissdSparseKvRunReq*,
                                        AissdSparseKvRunResp*, int);
using AissdRpcInitFn = int (*)();

struct AissdClientApi {
  void* handle = nullptr;
  AissdRunNativeExtentsFn run = nullptr;
};

AissdClientApi& get_aissd_client() {
  static AissdClientApi api;
  static std::once_flag once;
  std::call_once(once, []() {
    const char* lib = std::getenv("AISSD_SPARSE_KV_LIB");
    if (lib == nullptr || std::strlen(lib) == 0) {
      lib = "libaissd_sparse_kv_client.so";
    }
    api.handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) {
      throw std::runtime_error(std::string("dlopen AISSD_SPARSE_KV_LIB failed: ") + dlerror());
    }
    auto init = reinterpret_cast<AissdRpcInitFn>(dlsym(api.handle, "aissd_sparse_kv_rpc_init"));
    api.run = reinterpret_cast<AissdRunNativeExtentsFn>(dlsym(api.handle, "aissd_sparse_kv_run_native_extents"));
    if (!api.run) {
      throw std::runtime_error("dlsym aissd_sparse_kv_run_native_extents failed");
    }
    if (init) {
      const int rc = init();
      if (rc != 0) {
        throw std::runtime_error("aissd_sparse_kv_rpc_init failed rc=" + std::to_string(rc));
      }
    }
  });
  return api;
}

inline void aissd_check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

inline int64_t aissd_elem_i64_cpu(const at::Tensor& t, int64_t i) {
  TORCH_CHECK(!t.is_cuda(), "AISSD metadata tensor must be CPU after materialization");
  if (t.scalar_type() == at::kInt) return static_cast<int64_t>(t.data_ptr<int32_t>()[i]);
  if (t.scalar_type() == at::kLong) return t.data_ptr<int64_t>()[i];
  TORCH_CHECK(false, "AISSD metadata tensor must be int32/int64");
}

inline int64_t aissd_elem2_i64_cpu(const at::Tensor& t, int64_t i, int64_t j) {
  return aissd_elem_i64_cpu(t, i * t.size(1) + j);
}

inline uint64_t aissd_elem3_u64_cpu(const at::Tensor& t, int64_t a, int64_t b, int64_t c) {
  const int64_t idx = (a * t.size(1) + b) * t.size(2) + c;
  TORCH_CHECK(!t.is_cuda() && t.scalar_type() == at::kLong,
              "AISSD extent/shape tensor must be CPU int64");
  return static_cast<uint64_t>(t.data_ptr<int64_t>()[idx]);
}

uint32_t aissd_query_dtype_code(const at::Tensor& q) {
  if (q.scalar_type() == at::kFloat) return 1;
  if (q.scalar_type() == at::kHalf) return 2;
  if (q.scalar_type() == at::kBFloat16) return 3;
  if (q.scalar_type() == at::kChar) return 4;
  TORCH_CHECK(false, "AISSD selector unsupported query dtype");
}

uint64_t aissd_job_id(int64_t layer_id, int64_t req_index) {
  return (static_cast<uint64_t>(layer_id) << 32) ^ static_cast<uint64_t>(req_index + 1);
}

int64_t aissd_query_tail_tokens() {
  const char* v = std::getenv("AISSD_SPARSE_KV_QUERY_TAIL_TOKENS");
  if (!v || !v[0]) {
    return 1;
  }
  const long n = std::strtol(v, nullptr, 10);
  return n > 0 ? static_cast<int64_t>(n) : 1;
}

int64_t aissd_select_query_token_index(const at::Tensor& query,
                                       const at::Tensor& req_token_lens_cpu,
                                       int64_t req_index,
                                       int64_t reqs) {
  if (query.dim() < 3) {
    return 0;
  }
  const int64_t q_tokens = query.size(0);
  TORCH_CHECK(q_tokens > 0, "aissd_sparse_kv_select: empty query tensor");

  const int64_t tail_tokens = aissd_query_tail_tokens();
  TORCH_CHECK(tail_tokens == 1,
              "aissd_sparse_kv_select currently supports AISSD_SPARSE_KV_QUERY_TAIL_TOKENS=1, got ",
              tail_tokens);

  // Decode/common path: one query row per active request.
  if (q_tokens == reqs) {
    return req_index;
  }

  // Single active request with a prefill/chunked-prefill query batch.  Use the
  // last query row only; the SSD selector/NPU qK model is a decode-style
  // single-token selector.
  if (reqs == 1) {
    return q_tokens - 1;
  }

  // If req_token_lens describes the current query rows, use the last row of
  // each request span.
  int64_t total = 0;
  for (int64_t r = 0; r < reqs; ++r) {
    total += std::max<int64_t>(aissd_elem_i64_cpu(req_token_lens_cpu, r), 0);
  }
  if (total == q_tokens) {
    int64_t base = 0;
    for (int64_t r = 0; r < req_index; ++r) {
      base += std::max<int64_t>(aissd_elem_i64_cpu(req_token_lens_cpu, r), 0);
    }
    const int64_t len = std::max<int64_t>(aissd_elem_i64_cpu(req_token_lens_cpu, req_index), 1);
    return std::min<int64_t>(base + len - 1, q_tokens - 1);
  }

  // Conservative fallback for unusual batching metadata: select the last
  // available query token instead of copying the full prefill query tensor into
  // the CMB raw area.
  return q_tokens - 1;
}

at::Tensor aissd_make_single_query_for_req(const at::Tensor& query,
                                           const at::Tensor& req_token_lens_cpu,
                                           int64_t req_index,
                                           int64_t reqs) {
  TORCH_CHECK(query.dim() == 2 || query.dim() == 3,
              "aissd_sparse_kv_select: expected query shape [H,D] or [T,H,D], got dim=",
              query.dim());
  at::Tensor q_one;
  if (query.dim() == 2) {
    q_one = query;
  } else {
    const int64_t token_index = aissd_select_query_token_index(query, req_token_lens_cpu, req_index, reqs);
    q_one = query.select(0, token_index);
  }
  return q_one.contiguous();
}

void ensure_not_cuda_graph_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaError_t rc = cudaStreamIsCapturing(stream, &status);
  if (rc == cudaSuccess && status != cudaStreamCaptureStatusNone) {
    throw std::runtime_error(
        "aissd_sparse_kv_select performs HOST<->SSD RPC and must run outside CUDA graph capture. "
        "Split/exclude this op from vLLM CUDA graph capture; keep sparse_flash_attention itself graph-captured.");
  }
}

bool aissd_env_truthy(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !v[0]) return false;
  return std::strcmp(v, "0") != 0 &&
         std::strcmp(v, "false") != 0 &&
         std::strcmp(v, "False") != 0 &&
         std::strcmp(v, "no") != 0 &&
         std::strcmp(v, "off") != 0;
}

uint32_t aissd_env_u32(const char* name, uint32_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !v[0]) return fallback;
  char* endp = nullptr;
  const unsigned long x = std::strtoul(v, &endp, 0);
  if (endp == v) return fallback;
  return x > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(x);
}

uint64_t aissd_fnv1a64(const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

bool aissd_q_trace_reserve(int64_t layer_id, uint64_t* seq_out) {
  if (!aissd_env_truthy("AISSD_SPARSE_KV_Q_TRACE")) return false;
  const uint32_t layer_filter =
      aissd_env_u32("AISSD_SPARSE_KV_Q_TRACE_LAYER", 0u);
  if (layer_filter != UINT32_MAX &&
      layer_id != static_cast<int64_t>(layer_filter)) {
    return false;
  }
  static unsigned long long match_calls = 0;
  const unsigned long long seq =
      __sync_add_and_fetch(&match_calls, 1ULL);
  const uint32_t max_calls =
      aissd_env_u32("AISSD_SPARSE_KV_Q_TRACE_MAX_CALLS", 2u);
  if (max_calls != 0u && seq > static_cast<unsigned long long>(max_calls)) {
    return false;
  }
  if (seq_out) *seq_out = static_cast<uint64_t>(seq);
  return true;
}

void aissd_q_trace_dump(const char* stage,
                        const void* data,
                        size_t bytes,
                        uint32_t q_dtype,
                        int64_t layer_id,
                        uint64_t job_id,
                        uint64_t request_id,
                        uint64_t seq) {
  if (!data || bytes == 0) return;
  const auto* p = static_cast<const uint8_t*>(data);
  const size_t words = bytes / sizeof(uint16_t);
  size_t zero_u16 = 0;
  for (size_t i = 0; i < words; ++i) {
    uint16_t w = 0;
    std::memcpy(&w, p + i * sizeof(uint16_t), sizeof(w));
    if (w == 0) ++zero_u16;
  }

  const uint32_t show_cfg =
      aissd_env_u32("AISSD_SPARSE_KV_Q_TRACE_SHOW_U16", 16u);
  const size_t show = std::min<size_t>(words, show_cfg);

  std::fprintf(stderr,
               "[aissd-q-trace][%s] seq=%llu layer=%lld job_id=%llu "
               "request_id=%llu q_dtype=%u bytes=%zu fnv1a64=0x%016llx "
               "u16_words=%zu zero_u16=%zu first_u16=",
               stage ? stage : "unknown",
               static_cast<unsigned long long>(seq),
               static_cast<long long>(layer_id),
               static_cast<unsigned long long>(job_id),
               static_cast<unsigned long long>(request_id),
               q_dtype,
               bytes,
               static_cast<unsigned long long>(aissd_fnv1a64(data, bytes)),
               words,
               zero_u16);
  for (size_t i = 0; i < show; ++i) {
    uint16_t w = 0;
    std::memcpy(&w, p + i * sizeof(uint16_t), sizeof(w));
    std::fprintf(stderr, "%s0x%04x", i ? "," : "[",
                 static_cast<unsigned int>(w));
  }
  std::fprintf(stderr, "]\n");
  std::fflush(stderr);
}

}  // namespace

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
    int64_t timeout_ms) {
  TORCH_CHECK(query.is_cuda(), "aissd_sparse_kv_select: query must be CUDA");
  TORCH_CHECK(selected_block_table.is_cuda() && selected_block_lens.is_cuda() &&
                  selected_ready_flags.is_cuda() && fa_block_table.is_cuda() && fa_seq_lens.is_cuda(),
              "aissd_sparse_kv_select: output metadata tensors must be CUDA");
  TORCH_CHECK(selected_block_table.scalar_type() == at::kInt && selected_block_lens.scalar_type() == at::kInt &&
                  selected_ready_flags.scalar_type() == at::kInt && fa_block_table.scalar_type() == at::kInt &&
                  fa_seq_lens.scalar_type() == at::kInt,
              "aissd_sparse_kv_select: selected/fa metadata tensors must be int32");
  TORCH_CHECK(selected_block_table.dim() == 2 && fa_block_table.dim() == 2 &&
                  selected_block_table.size(0) == fa_block_table.size(0) &&
                  selected_block_table.size(1) == fa_block_table.size(1),
              "aissd_sparse_kv_select: fa_block_table shape must match selected_block_table");
  TORCH_CHECK(selected_block_lens.numel() == fa_seq_lens.numel(),
              "aissd_sparse_kv_select: fa_seq_lens length must match selected_block_lens");

  // Preserve the SSD RPC result itself for Phase-1 quality tracing. Do not
  // infer selected chunks back from selected_block_table: block IDs are an
  // attention representation and can be ambiguous/reindexed independently of
  // the SSD selector's chunk IDs.
  TORCH_CHECK(!aissd_rpc_selected_chunk_ids.is_cuda() &&
                  !aissd_rpc_selected_chunk_lens.is_cuda(),
              "aissd_sparse_kv_select: RPC selected chunk outputs must be CPU tensors");
  TORCH_CHECK(aissd_rpc_selected_chunk_ids.scalar_type() == at::kInt &&
                  aissd_rpc_selected_chunk_lens.scalar_type() == at::kInt,
              "aissd_sparse_kv_select: RPC selected chunk outputs must be int32");
  TORCH_CHECK(aissd_rpc_selected_chunk_ids.dim() == 2 &&
                  aissd_rpc_selected_chunk_lens.dim() == 1 &&
                  aissd_rpc_selected_chunk_ids.size(0) ==
                      aissd_rpc_selected_chunk_lens.size(0),
              "aissd_sparse_kv_select: RPC selected chunk output shape mismatch");
  TORCH_CHECK(aissd_rpc_selected_chunk_ids.is_contiguous() &&
                  aissd_rpc_selected_chunk_lens.is_contiguous(),
              "aissd_sparse_kv_select: RPC selected chunk outputs must be contiguous");

  auto* rpc_selected_ids = aissd_rpc_selected_chunk_ids.data_ptr<int32_t>();
  auto* rpc_selected_lens = aissd_rpc_selected_chunk_lens.data_ptr<int32_t>();
  std::fill(rpc_selected_ids,
            rpc_selected_ids + aissd_rpc_selected_chunk_ids.numel(), -1);
  std::fill(rpc_selected_lens,
            rpc_selected_lens + aissd_rpc_selected_chunk_lens.numel(), 0);

  const c10::cuda::CUDAGuard guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ensure_not_cuda_graph_capturing(stream);

  auto active_cpu = active_reqs.cpu();
  const int64_t reqs = aissd_elem_i64_cpu(active_cpu, 0);
  if (reqs <= 0 || backend == 0) {
    return;
  }
  TORCH_CHECK(reqs <= selected_block_table.size(0),
              "aissd_sparse_kv_select: active_reqs exceeds metadata rows");
  TORCH_CHECK(reqs <= aissd_rpc_selected_chunk_ids.size(0),
              "aissd_sparse_kv_select: active_reqs exceeds RPC selected chunk rows");

  // Materialize CPU metadata. LMCache normally prepares these on CPU; .cpu()
  // also keeps the call robust if a future path publishes pinned/CUDA tensors.
  auto req_lens_cpu = req_token_lens.cpu();
  auto cand_count_cpu = aissd_candidate_count.cpu();
  auto chunk_ids_cpu = aissd_candidate_chunk_ids.cpu();
  auto block_ids_cpu = aissd_candidate_block_ids.cpu();
  auto block_lens_cpu = aissd_candidate_block_lens.cpu();
  auto tok_start_cpu = aissd_candidate_token_start.cpu();
  auto tok_end_cpu = aissd_candidate_token_end.cpu();
  auto dtype_cpu = aissd_candidate_dtype.cpu();
  auto fmt_cpu = aissd_candidate_fmt.cpu();
  auto ndim_cpu = aissd_candidate_ndim.cpu();
  auto shape_cpu = aissd_candidate_shape.cpu();
  auto ext_count_cpu = aissd_candidate_extent_count.cpu();
  auto ext_lba_cpu = aissd_candidate_extent_lba.cpu();
  auto ext_bytes_cpu = aissd_candidate_extent_bytes.cpu();

  AissdClientApi& api = get_aissd_client();
  const int64_t max_blocks = selected_block_table.size(1);
  std::vector<int32_t> out_table(selected_block_table.numel(), -1);
  std::vector<int32_t> out_lens(selected_block_lens.numel(), 0);
  std::vector<int32_t> out_ready(selected_ready_flags.numel(), 0);
  std::vector<int32_t> out_fa_table(fa_block_table.numel(), 0);
  std::vector<int32_t> out_fa_lens(fa_seq_lens.numel(), 1);

  for (int64_t r = 0; r < reqs; ++r) {
    const int64_t cand_n = aissd_elem_i64_cpu(cand_count_cpu, r);
    TORCH_CHECK(cand_n > 0, "aissd_sparse_kv_select: candidate_count is zero");
    TORCH_CHECK(cand_n <= AISSD_MAX_CANDIDATES,
                "aissd_sparse_kv_select: candidate_count exceeds protocol max");

    at::Tensor q_one = aissd_make_single_query_for_req(query, req_lens_cpu, r, reqs);
    TORCH_CHECK(q_one.dim() == 2 && q_one.size(0) == num_q_heads && q_one.size(1) == head_dim,
                "aissd_sparse_kv_select: selected q tail shape mismatch, got ",
                q_one.sizes(), " expected [", num_q_heads, ",", head_dim, "]");
    const int64_t q_bytes_i64 = q_one.numel() * q_one.element_size();
    TORCH_CHECK(q_bytes_i64 > 0 && q_bytes_i64 <= UINT32_MAX,
                "aissd_sparse_kv_select: invalid single-token q_bytes=", q_bytes_i64);
    std::vector<uint8_t> q_host(static_cast<size_t>(q_bytes_i64));
    aissd_check_cuda(cudaMemcpyAsync(q_host.data(), q_one.data_ptr(), q_bytes_i64,
                                     cudaMemcpyDeviceToHost, stream),
                     "aissd_sparse_kv_select: copy q tail D2H");
    aissd_check_cuda(cudaStreamSynchronize(stream),
                     "aissd_sparse_kv_select: sync q tail D2H");
    const uint32_t q_dtype = aissd_query_dtype_code(q_one);
    const uint32_t q_bytes = static_cast<uint32_t>(q_bytes_i64);

    AissdSparseKvRunReq req{};
    AissdSparseKvRunResp resp{};
    req.cmd = CMD_SPARSE_KV_RUN_MANIFEST_LBA;
    req.version = AISSD_SPARSE_KV_PROTOCOL_VERSION;
    req.job_id = aissd_job_id(layer_id, r);
    req.request_id = static_cast<uint64_t>(r + 1);
    req.layer_id = static_cast<uint32_t>(layer_id);
    req.backend = static_cast<uint32_t>(backend);
    req.num_q_heads = static_cast<uint32_t>(num_q_heads);
    req.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    req.head_dim = static_cast<uint32_t>(head_dim);
    req.chunk_size = static_cast<uint32_t>(chunk_size);
    req.block_size = static_cast<uint32_t>(block_size);
    req.top_n_chunks = static_cast<uint32_t>(top_n_chunks);
    req.top_m = static_cast<uint32_t>(top_m);
    req.score_mode = static_cast<uint32_t>(score_mode);
    req.q_dtype = q_dtype;
    req.kv_dtype = static_cast<uint32_t>(aissd_elem2_i64_cpu(dtype_cpu, r, 0));
    req.q_token_count = 1;
    req.candidate_chunk_count = static_cast<uint32_t>(cand_n);
    req.q_bytes = static_cast<uint32_t>(q_bytes);
    req.manifest_block_size = static_cast<uint32_t>(manifest_block_size);
    req.flags = AISSD_SPARSE_KV_FLAG_Q_INLINE_CMB |
                AISSD_SPARSE_KV_FLAG_NATIVE_EXTENTS |
                AISSD_SPARSE_KV_FLAG_RESULT_INLINE_CMB;

    for (int64_t c = 0; c < cand_n; ++c) {
      auto& dst = req.candidates[c];
      dst.chunk_index = static_cast<uint32_t>(aissd_elem2_i64_cpu(chunk_ids_cpu, r, c));
      dst.token_start = static_cast<uint32_t>(aissd_elem2_i64_cpu(tok_start_cpu, r, c));
      dst.token_end = static_cast<uint32_t>(aissd_elem2_i64_cpu(tok_end_cpu, r, c));
      dst.num_tokens = dst.token_end > dst.token_start
                           ? dst.token_end - dst.token_start
                           : static_cast<uint32_t>(chunk_size);
      dst.dtype = static_cast<uint32_t>(aissd_elem2_i64_cpu(dtype_cpu, r, c));
      dst.fmt = static_cast<uint32_t>(aissd_elem2_i64_cpu(fmt_cpu, r, c));
      dst.ndim = static_cast<uint32_t>(aissd_elem2_i64_cpu(ndim_cpu, r, c));
      for (uint32_t d = 0; d < AISSD_MAX_DIMS; ++d) {
        dst.shape[d] = static_cast<uint32_t>(aissd_elem3_u64_cpu(shape_cpu, r, c, d));
      }
      const int64_t extent_n = aissd_elem2_i64_cpu(ext_count_cpu, r, c);
      TORCH_CHECK(extent_n > 0 && extent_n <= AISSD_MAX_EXTENTS_PER_CANDIDATE,
                  "aissd_sparse_kv_select: invalid candidate extent_count");
      dst.extent_count = static_cast<uint32_t>(extent_n);
      uint64_t nbytes = 0;
      for (int64_t e = 0; e < extent_n; ++e) {
        dst.extents[e].lba = aissd_elem3_u64_cpu(ext_lba_cpu, r, c, e);
        dst.extents[e].bytes = aissd_elem3_u64_cpu(ext_bytes_cpu, r, c, e);
        nbytes += dst.extents[e].bytes;
      }
      dst.nbytes = nbytes;
    }

    uint64_t q_trace_seq = 0;
    if (aissd_q_trace_reserve(layer_id, &q_trace_seq)) {
      aissd_q_trace_dump("HOST-D2H-BEFORE-RPC",
                         q_host.data(),
                         static_cast<size_t>(q_bytes),
                         q_dtype,
                         layer_id,
                         req.job_id,
                         req.request_id,
                         q_trace_seq);
    }

    const int rc = api.run(q_host.data(), static_cast<uint32_t>(q_bytes), &req, &resp,
                           static_cast<int>(timeout_ms));
    TORCH_CHECK(rc == 0 && resp.status == 0,
                "aissd_sparse_kv_run_native_extents failed rc=", rc,
                " status=", resp.status, " error_code=", resp.error_code);

    const int64_t rpc_capacity = aissd_rpc_selected_chunk_ids.size(1);
    TORCH_CHECK(static_cast<int64_t>(resp.selected_chunk_count) <= rpc_capacity,
                "aissd_sparse_kv_select: SSD returned selected_chunk_count=",
                resp.selected_chunk_count, " larger than trace capacity=", rpc_capacity);
    rpc_selected_lens[r] = static_cast<int32_t>(resp.selected_chunk_count);
    for (uint32_t si = 0; si < resp.selected_chunk_count; ++si) {
      rpc_selected_ids[r * rpc_capacity + static_cast<int64_t>(si)] =
          static_cast<int32_t>(resp.selected_chunk_ids[si]);
    }

    int64_t out_len = 0;
    for (uint32_t si = 0; si < resp.selected_chunk_count && out_len < max_blocks; ++si) {
      const uint32_t chunk_id = resp.selected_chunk_ids[si];
      for (int64_t c = 0; c < cand_n && out_len < max_blocks; ++c) {
        if (static_cast<uint32_t>(aissd_elem2_i64_cpu(chunk_ids_cpu, r, c)) != chunk_id) continue;
        const int64_t blen = aissd_elem2_i64_cpu(block_lens_cpu, r, c);
        for (int64_t b = 0; b < blen && out_len < max_blocks; ++b) {
          const int64_t idx = (r * aissd_candidate_block_ids.size(1) + c) *
                              aissd_candidate_block_ids.size(2) + b;
          const int32_t block_id = block_ids_cpu.data_ptr<int32_t>()[idx];
          if (block_id >= 0) {
            out_table[r * max_blocks + out_len] = block_id;
            out_fa_table[r * max_blocks + out_len] = block_id;
            ++out_len;
          }
        }
      }
    }
    TORCH_CHECK(out_len > 0, "aissd_sparse_kv_select: SSD selected zero KV blocks");
    out_lens[r] = static_cast<int32_t>(out_len);
    out_ready[r] = 1;
    out_fa_lens[r] = static_cast<int32_t>(out_len * block_size);
  }

  aissd_check_cuda(cudaMemcpyAsync(selected_block_table.data_ptr(), out_table.data(),
                                   selected_block_table.numel() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "aissd_sparse_kv_select: copy selected_block_table H2D");
  aissd_check_cuda(cudaMemcpyAsync(selected_block_lens.data_ptr(), out_lens.data(),
                                   selected_block_lens.numel() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "aissd_sparse_kv_select: copy selected_block_lens H2D");
  aissd_check_cuda(cudaMemcpyAsync(selected_ready_flags.data_ptr(), out_ready.data(),
                                   selected_ready_flags.numel() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "aissd_sparse_kv_select: copy selected_ready_flags H2D");
  aissd_check_cuda(cudaMemcpyAsync(fa_block_table.data_ptr(), out_fa_table.data(),
                                   fa_block_table.numel() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "aissd_sparse_kv_select: copy fa_block_table H2D");
  aissd_check_cuda(cudaMemcpyAsync(fa_seq_lens.data_ptr(), out_fa_lens.data(),
                                   fa_seq_lens.numel() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "aissd_sparse_kv_select: copy fa_seq_lens H2D");
}


TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sparse_flash_attention", &sparse_flash_attention);
  m.impl("aissd_sparse_kv_select", &aissd_sparse_kv_select);
}
