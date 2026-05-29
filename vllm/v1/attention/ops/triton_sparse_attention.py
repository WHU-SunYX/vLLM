# SPDX-License-Identifier: Apache-2.0
"""Naive sparse paged-attention fallback for AI-SSD selected KV chunks.

This module intentionally avoids Triton-specific code for the first functional
version. It provides a torch implementation with the same public entry point so
we can validate metadata and correctness before replacing it with a real Triton
kernel.
"""

from typing import Any

import torch


def _as_token_head_dim(x: torch.Tensor) -> torch.Tensor:
    """Normalize a KV cache block to [tokens, heads, head_dim]."""
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D KV block, got shape={tuple(x.shape)}")
    # Common layouts:
    #   [block_size, num_heads, head_dim]  (NHD)
    #   [num_heads, block_size, head_dim]  (HND)
    # Head count is usually <= 128 and block size is commonly 16/32/128. When
    # ambiguous we keep the original order, which is the default NHD path.
    if x.shape[0] <= 128 and x.shape[1] > x.shape[0] and x.shape[2] >= 16:
        return x.transpose(0, 1).contiguous()
    return x


def _get_kv_from_cache(kv_cache: Any, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one block from vLLM-style paged KV cache.

    Supported forms:
      - tuple/list: (key_cache, value_cache)
      - tensor [2, num_blocks, block, kv_heads, head]
      - tensor [num_blocks, 2, block, kv_heads, head]
      - tensor [num_blocks, block, kv_heads, head] (K=V fallback for debugging)
    """
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("kv_cache tuple/list must contain key and value caches")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        return _as_token_head_dim(k_cache[block_id]), _as_token_head_dim(v_cache[block_id])

    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError(f"Unsupported kv_cache type: {type(kv_cache).__name__}")

    if kv_cache.ndim >= 5 and kv_cache.shape[0] == 2:
        return (
            _as_token_head_dim(kv_cache[0, block_id]),
            _as_token_head_dim(kv_cache[1, block_id]),
        )
    if kv_cache.ndim >= 5 and kv_cache.shape[1] == 2:
        return (
            _as_token_head_dim(kv_cache[block_id, 0]),
            _as_token_head_dim(kv_cache[block_id, 1]),
        )
    if kv_cache.ndim == 4:
        # Debug-only fallback when only one cache tensor is supplied.
        block = _as_token_head_dim(kv_cache[block_id])
        return block, block

    raise ValueError(f"Unsupported kv_cache shape: {tuple(kv_cache.shape)}")


def _repeat_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Repeat KV heads for GQA/MQA so x becomes [tokens, num_q_heads, head]."""
    kv_heads = x.shape[1]
    if kv_heads == num_q_heads:
        return x
    if num_q_heads % kv_heads != 0:
        raise ValueError(f"Cannot repeat kv_heads={kv_heads} to q_heads={num_q_heads}")
    repeat = num_q_heads // kv_heads
    return x.repeat_interleave(repeat, dim=1)


def triton_sparse_paged_attention(
    query: torch.Tensor,
    kv_cache: Any,
    sparse_block_table: torch.Tensor,
    sparse_block_lens: torch.Tensor,
    scale: float,
    **kwargs: Any,
) -> torch.Tensor:
    """Torch fallback for selected-block sparse attention.

    Args:
      query: [num_query_tokens, num_heads, head_dim]
      kv_cache: vLLM paged KV cache. See _get_kv_from_cache().
      sparse_block_table: [num_reqs, max_selected_blocks], -1 padded block ids.
      sparse_block_lens: [num_reqs], valid block count per request.
      scale: attention scale, usually 1/sqrt(head_dim).

    Optional kwargs:
      query_start_loc: [num_reqs + 1] start offsets for query tokens. If omitted,
        the function assumes either one query per request or a single request.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be [tokens, heads, head_dim], got {tuple(query.shape)}")
    if sparse_block_table is None or sparse_block_lens is None:
        raise ValueError("sparse_block_table and sparse_block_lens are required")

    device = query.device
    dtype = query.dtype
    num_tokens, num_heads, _ = query.shape
    out = torch.zeros_like(query)

    table = sparse_block_table.to(device="cpu", dtype=torch.long)
    lens = sparse_block_lens.to(device="cpu", dtype=torch.long)
    num_reqs = int(table.shape[0])

    query_start_loc = kwargs.get("query_start_loc", None)
    if isinstance(query_start_loc, torch.Tensor) and query_start_loc.numel() >= num_reqs + 1:
        q_starts = query_start_loc.to(device="cpu", dtype=torch.long).tolist()
    elif num_reqs == num_tokens:
        q_starts = list(range(num_reqs + 1))
    else:
        q_starts = [0, num_tokens] + [num_tokens] * max(0, num_reqs - 1)

    for req_idx in range(num_reqs):
        q_start = int(q_starts[req_idx])
        q_end = int(q_starts[req_idx + 1]) if req_idx + 1 < len(q_starts) else num_tokens
        if q_end <= q_start:
            continue
        block_count = int(lens[req_idx].item())
        if block_count <= 0:
            continue
        block_ids = [int(x) for x in table[req_idx, :block_count].tolist() if int(x) >= 0]
        if not block_ids:
            continue

        k_blocks = []
        v_blocks = []
        for block_id in block_ids:
            k_block, v_block = _get_kv_from_cache(kv_cache, block_id)
            k_blocks.append(k_block.to(device=device, dtype=dtype))
            v_blocks.append(v_block.to(device=device, dtype=dtype))
        k = _repeat_kv_heads(torch.cat(k_blocks, dim=0), num_heads)
        v = _repeat_kv_heads(torch.cat(v_blocks, dim=0), num_heads)

        q = query[q_start:q_end]
        # scores: [q_tokens, heads, kv_tokens]
        scores = torch.einsum("qhd,khd->qhk", q, k) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(dtype)
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", probs, v)

    return out
