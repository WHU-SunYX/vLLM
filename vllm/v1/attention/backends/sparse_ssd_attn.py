# SPDX-License-Identifier: Apache-2.0
"""AI-SSD sparse attention backend for vLLM V1.

This backend is wired into vLLM's normal attention backend selection path.
NO_PYTHON_ATTENTION_HOOK: runtime sparse attention must be driven by the step context prepared by LMCacheConnector.start_load_kv()/prepare_sparse_kv_step(), not by Attention.forward Python hooks.
For production, it must not perform Python-side Q-aware selection or GDS IO
inside ``forward``: CUDA graph replay will not re-enter Python for every real
request.  Instead, LMCache prepares a step-level sparse KV context in
KVConnector.pre_forward(), and this backend calls a C++/CUDA custom op that
consumes the real per-layer Q tensor plus that step context.

If the custom op is not registered, this backend fails fast when full KV load is
disabled, and can optionally fall back to FlashAttention only in debug/dry-run
mode where full KV remains available.
"""

from typing import Any
import copy
import os
import time

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
try:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
except Exception:  # pragma: no cover - depends on vLLM/flash-attn build
    flash_attn_varlen_func = None
from vllm import _custom_ops as vllm_ops

logger = init_logger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).lower() not in ("0", "false", "no", "off")


def _sparse_debug_counters_enabled() -> bool:
    # CUDA counter readback can synchronize GPU work; keep disabled by default.
    return _env_flag("VLLM_SPARSE_ATTN_DEBUG_COUNTERS", "0")


def _sparse_kv_debug_enabled() -> bool:
    # High-frequency Python sparse-attention route logs.
    return _env_flag("VLLM_SPARSE_KV_DEBUG", "0")


def _sparse_ctx_ptr_debug_enabled() -> bool:
    # data_ptr diagnostics for CUDA graph context lifetime debugging.
    return _env_flag("VLLM_SPARSE_CTX_PTR_DEBUG", "0")


def _sparse_attention_impl() -> str:
    # custom: current self-written sparse_flash_attention CUDA op.
    # fa_varlen: experimental path that treats selected blocks as a compacted
    # FlashAttention paged block_table and reuses flash_attn_varlen_func.
    return str(os.environ.get("VLLM_SPARSE_ATTENTION_IMPL", "fa_varlen")).lower()


def _sparse_fa_replay_debug_enabled() -> bool:
    # Lightweight CUDA-graph replay marker for FA-varlen sparse attention.
    # Disabled by default because draining the device marker synchronizes.
    return _env_flag("VLLM_SPARSE_FA_REPLAY_DEBUG", "0")


def _sparse_attention_diag_enabled() -> bool:
    return _env_flag("VLLM_SPARSE_ATTENTION_DIAG", "0")


def _sparse_attention_profile_enabled() -> bool:
    # High-overhead profiling: uses CUDA events and, by default, synchronizes.
    # Enable only for profiling runs, not for final throughput measurement.
    return _env_flag("VLLM_SPARSE_ATTN_PROFILE", "0") or _env_flag(
        "VLLM_SPARSE_ATTN_TIME", "0"
    )


def _sparse_attention_profile_sync_enabled() -> bool:
    # CUDA event elapsed_time is only reliable after synchronization.  Keep this
    # enabled for accurate profiling; disable only if you want enqueue-time logs.
    return _env_flag("VLLM_SPARSE_ATTN_PROFILE_SYNC", "1")


def _sparse_attention_nvtx_enabled() -> bool:
    return _env_flag("VLLM_SPARSE_ATTN_NVTX", "0")


def _sparse_profile_begin(tag: str, layer_name: str, query: Any | None = None) -> dict[str, Any] | None:
    if not (_sparse_attention_profile_enabled() or _sparse_attention_nvtx_enabled()):
        return None
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        capturing = False

    prof: dict[str, Any] = {
        "tag": tag,
        "layer_name": layer_name,
        "cpu_t0": time.perf_counter(),
        "cuda_start": None,
        "cuda_end": None,
        "nvtx": False,
    }
    if _sparse_attention_nvtx_enabled():
        try:
            torch.cuda.nvtx.range_push(f"aissd:{tag}:{layer_name}")
            prof["nvtx"] = True
        except Exception:
            prof["nvtx"] = False

    if (
        _sparse_attention_profile_enabled()
        and not capturing
        and isinstance(query, torch.Tensor)
        and query.is_cuda
        and torch.cuda.is_available()
    ):
        try:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            prof["cuda_start"] = ev0
            prof["cuda_end"] = ev1
        except Exception:
            prof["cuda_start"] = None
            prof["cuda_end"] = None
    return prof


def _sparse_profile_end(
    prof: dict[str, Any] | None,
    step_context: dict[str, Any] | None,
    query: Any | None = None,
    impl: str | None = None,
    extra: str = "",
) -> None:
    if prof is None:
        return
    cpu_ms = (time.perf_counter() - float(prof.get("cpu_t0", time.perf_counter()))) * 1000.0
    cuda_ms: float | None = None
    ev0 = prof.get("cuda_start")
    ev1 = prof.get("cuda_end")
    if ev0 is not None and ev1 is not None:
        try:
            ev1.record()
            if _sparse_attention_profile_sync_enabled():
                torch.cuda.synchronize()
                cuda_ms = float(ev0.elapsed_time(ev1))
        except Exception:
            cuda_ms = None
    if prof.get("nvtx"):
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass
    if not _sparse_attention_profile_enabled():
        return

    q_shape = tuple(query.shape) if isinstance(query, torch.Tensor) else None
    generation = None if step_context is None else step_context.get("context_generation")
    host_reqs = None if step_context is None else step_context.get("host_active_reqs")
    selected_blocks = None if step_context is None else step_context.get("host_selected_blocks")
    top_n = None if step_context is None else step_context.get("top_n_chunks")
    chunk_size = None if step_context is None else step_context.get("chunk_size")
    block_size = None if step_context is None else step_context.get("block_size")
    candidate_count = None if step_context is None else step_context.get("aissd_candidate_count")
    if isinstance(candidate_count, torch.Tensor):
        # Do not read CUDA tensors here; logging a tensor object is enough and avoids extra sync.
        candidate_count_repr = f"Tensor(shape={tuple(candidate_count.shape)}, device={candidate_count.device})"
    else:
        candidate_count_repr = candidate_count

    logger.info(
        "[sparse-attn-time] tag=%s impl=%s layer=%s generation=%s "
        "q_shape=%s host_reqs=%s selected_blocks=%s candidate_count=%s "
        "top_n=%s chunk=%s block=%s cpu_ms=%.3f cuda_ms=%s sync=%s%s%s",
        prof.get("tag"),
        impl,
        prof.get("layer_name"),
        generation,
        q_shape,
        host_reqs,
        selected_blocks,
        candidate_count_repr,
        top_n,
        chunk_size,
        block_size,
        cpu_ms,
        "NA" if cuda_ms is None else f"{cuda_ms:.3f}",
        _sparse_attention_profile_sync_enabled(),
        " " if extra else "",
        extra,
    )


def _aissd_selector_stats_enabled() -> bool:
    return _env_flag("AISSD_SPARSE_KV_SELECTOR_STATS", "0")


def _aissd_sparse_kv_e2e_stats_enabled(
    step_context: dict[str, Any] | None = None,
) -> bool:
    # Sparse KV E2E bandwidth/breakdown summary.  Keep
    # AISSD_SPARSE_KV_SELECTOR_STATS as a compatibility alias, but allow the
    # low-volume E2E summary to be enabled without high-volume selector logs.
    if _env_flag("AISSD_SPARSE_KV_E2E_STATS", "0") or _aissd_selector_stats_enabled():
        return True
    if step_context is not None:
        return bool(step_context.get("sparse_kv_e2e_stats_enabled", False))
    return False


def _aissd_layer_reuse_enabled() -> bool:
    # Disable with AISSD_SPARSE_KV_LAYER_REUSE=0 for exact per-layer selection.
    # When enabled, AISSD_LAYER_REUSE_STRATEGY controls the reuse policy.
    return _env_flag("AISSD_SPARSE_KV_LAYER_REUSE", "1")


def _aissd_layer_reuse_strategy() -> str:
    # global: legacy behavior, run selector once per decode step and reuse it
    # across all layers.
    # static: IndexCache-style training-free policy.  Run selector only on
    # AISSD_F_LAYERS and let other layers reuse the nearest previous F layer.
    return str(os.environ.get("AISSD_LAYER_REUSE_STRATEGY", "global")).strip().lower()


def _aissd_static_f_layers() -> tuple[int, ...]:
    raw = str(os.environ.get("AISSD_F_LAYERS", "0,6,12,18,24,30,35")).strip()
    layers: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layer_id = int(item)
        except ValueError:
            logger.warning(
                "Ignoring invalid AISSD_F_LAYERS entry %r in %r",
                item,
                raw,
            )
            continue
        if layer_id < 0:
            logger.warning(
                "Ignoring negative AISSD_F_LAYERS entry %r in %r",
                item,
                raw,
            )
            continue
        layers.append(layer_id)
    if not layers:
        layers = [0]
    return tuple(sorted(set(layers)))


def _aissd_parse_layer_id(layer_name: str, step_context: dict[str, Any]) -> int:
    layer_id = int(step_context.get("current_layer_id", -1))
    if layer_id >= 0:
        return layer_id
    try:
        import re

        match = re.search(r"layers\.(\d+)", str(layer_name))
        return int(match.group(1)) if match else 0
    except Exception:
        return 0


def _aissd_static_reuse_source_layer_id(layer_id: int, f_layers: tuple[int, ...]) -> int:
    # Nearest previous F layer.  If the first visible layer is before the first
    # configured F layer, fall back to the first configured F layer.
    source = f_layers[0]
    for f_layer in f_layers:
        if f_layer <= layer_id:
            source = f_layer
        else:
            break
    return int(source)


def _aissd_env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)


def _aissd_env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return float(default)


def _aissd_token_reuse_strategy() -> str:
    # legacy keeps the old behavior: selected metadata is reusable as long as
    # context_generation matches.  It is intentionally retained as a fast
    # fallback for experiments.
    return str(os.environ.get("AISSD_TOKEN_REUSE_STRATEGY", "none")).strip().lower()


def _aissd_token_reuse_is_legacy(strategy: str | None = None) -> bool:
    if strategy is None:
        strategy = _aissd_token_reuse_strategy()
    return strategy in ("legacy", "long", "long_reuse")


def _aissd_token_reuse_is_q_drift(strategy: str | None = None) -> bool:
    if strategy is None:
        strategy = _aissd_token_reuse_strategy()
    return strategy in ("q_drift", "q-drift", "qdrift")


def _aissd_token_reuse_interval() -> int:
    return max(1, _aissd_env_int("AISSD_TOKEN_REUSE_INTERVAL", 1))


def _aissd_token_reuse_max_staleness() -> int:
    value = _aissd_env_int("AISSD_TOKEN_REUSE_MAX_STALENESS", 0)
    if value <= 0:
        value = _aissd_token_reuse_interval()
    return max(1, value)


def _aissd_token_reuse_debug_enabled() -> bool:
    return _env_flag("AISSD_TOKEN_REUSE_DEBUG", "0")


def _aissd_q_drift_reduce() -> str:
    """Return the statistic used by q_drift to make the reuse decision.

    p95 is the default for the current experimental policy because a global
    request x head maximum was observed to reject every adjacent-token sample.
    Set AISSD_TOKEN_REUSE_Q_REDUCE=max to restore the original conservative
    maximum-based behavior.
    """
    value = str(os.environ.get("AISSD_TOKEN_REUSE_Q_REDUCE", "p95")).strip().lower()
    if value in ("max", "maximum", "amax"):
        return "max"
    if value in ("p95", "95", "percentile95", "quantile95"):
        return "p95"
    return "p95"


def _aissd_q_drift_cos_threshold() -> float:
    # Initial p95 threshold derived from the short Q-drift distribution run.
    return max(0.0, _aissd_env_float("AISSD_TOKEN_REUSE_Q_COS_THRESHOLD", 0.70))


def _aissd_q_drift_rel_l2_threshold() -> float:
    # Initial p95 threshold derived from the short Q-drift distribution run.
    return max(0.0, _aissd_env_float("AISSD_TOKEN_REUSE_Q_REL_L2_THRESHOLD", 1.20))


def _aissd_q_drift_max_staleness() -> int:
    # Start with one-token reuse only: age=1 may reuse, age>=2 refreshes the
    # anchor. Increase this only after measuring age=2/4 drift and top-n overlap.
    value = _aissd_env_int("AISSD_TOKEN_REUSE_Q_DRIFT_MAX_STALENESS", 0)
    if value <= 0:
        value = _aissd_env_int("AISSD_TOKEN_REUSE_MAX_STALENESS", 0)
    if value <= 0:
        value = 2
    return max(1, int(value))


def _aissd_token_reuse_overhead_log_enabled() -> bool:
    return _env_flag("AISSD_TOKEN_REUSE_OVERHEAD_LOG", "0")


def _aissd_tensor_int_list(value: Any, limit: int = 64) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        try:
            data = value.detach()
            if data.is_cuda:
                data = data.cpu()
            flat = data.reshape(-1)[:limit].tolist()
            return tuple(int(x) for x in flat)
        except Exception:
            return ()
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(x) for x in list(value)[:limit])
        except Exception:
            return ()
    return ()


def _aissd_q_drift_query_view(
    query: torch.Tensor,
    host_active_reqs: int,
    num_heads: int,
    head_size: int,
) -> torch.Tensor | None:
    """Return the active decode Q rows as [request, head, head_dim].

    The initial implementation assumes one decode query row per active request.
    CUDA-graph padding is tolerated by slicing the leading active rows.
    """
    if not isinstance(query, torch.Tensor) or query.dim() <= 0:
        return None
    active_reqs = int(host_active_reqs)
    heads = int(num_heads)
    dim = int(head_size)
    if active_reqs <= 0 or heads <= 0 or dim <= 0:
        return None
    if int(query.shape[0]) < active_reqs:
        return None
    active = query[:active_reqs]
    expected = active_reqs * heads * dim
    if int(active.numel()) != expected:
        return None
    return active.reshape(active_reqs, heads, dim)


def _aissd_current_candidate_signature(
    step_context: dict[str, Any],
) -> tuple[int, ...]:
    value = step_context.get("aissd_candidate_signature")
    active_reqs = max(0, int(step_context.get("host_active_reqs", 0) or 0))
    if isinstance(value, torch.Tensor):
        try:
            data = value.detach()
            if data.is_cuda:
                data = data.cpu()
            return tuple(int(x) for x in data.reshape(-1)[:active_reqs].tolist())
        except Exception:
            return ()
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(x) for x in list(value)[:active_reqs])
        except Exception:
            return ()
    return ()


def _aissd_q_drift_metric_stats(
    values: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    """Summarize one [request, head] drift tensor for threshold tuning.

    This function is called only when AISSD_TOKEN_REUSE_OVERHEAD_LOG=1.  The
    tensor is small (active requests x Q heads), so copying it to CPU keeps the
    production decision path simple and makes percentile values deterministic.
    """
    cpu = values.detach().to(device="cpu", dtype=torch.float32)
    if cpu.dim() != 2 or cpu.numel() == 0:
        return {}

    flat = cpu.reshape(-1)
    req_max = torch.amax(cpu, dim=-1)
    quantile_points = torch.tensor(
        [0.50, 0.90, 0.95, 0.99], dtype=torch.float32
    )
    p50, p90, p95, p99 = (
        float(x) for x in torch.quantile(flat, quantile_points).tolist()
    )
    req_p50, req_p95 = (
        float(x)
        for x in torch.quantile(
            req_max, torch.tensor([0.50, 0.95], dtype=torch.float32)
        ).tolist()
    )

    worst_flat = int(torch.argmax(flat).item())
    heads = int(cpu.shape[1])
    return {
        "mean": float(torch.mean(flat).item()),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": float(torch.amax(flat).item()),
        "exceed_ratio": float(torch.mean((flat > float(threshold)).float()).item()),
        "reqmax_mean": float(torch.mean(req_max).item()),
        "reqmax_p50": req_p50,
        "reqmax_p95": req_p95,
        "reqmax_max": float(torch.amax(req_max).item()),
        "req_exceed_ratio": float(
            torch.mean((req_max > float(threshold)).float()).item()
        ),
        "worst_req": int(worst_flat // heads),
        "worst_head": int(worst_flat % heads),
    }


def _aissd_compute_q_drift(
    current_q: torch.Tensor,
    anchor_q: torch.Tensor,
    *,
    collect_distribution: bool = False,
) -> tuple[float, float, float, float, dict[str, Any] | None]:
    """Return decision drift, maxima, and optional diagnostic distributions.

    AISSD_TOKEN_REUSE_Q_REDUCE selects the statistic used by the live decision:
    ``p95`` computes the 95th percentile over all active request x head samples,
    while ``max`` preserves the original global-maximum policy. Full mean/p50/
    p90/p95/p99 diagnostics are collected only when overhead logging is enabled.
    """
    if not isinstance(current_q, torch.Tensor) or not isinstance(anchor_q, torch.Tensor):
        return float("inf"), float("inf"), float("inf"), float("inf"), None
    if tuple(current_q.shape) != tuple(anchor_q.shape):
        return float("inf"), float("inf"), float("inf"), float("inf"), None

    q = current_q.detach().float()
    a = anchor_q.detach().to(device=q.device, dtype=torch.float32)
    dot = torch.sum(q * a, dim=-1)
    q_norm = torch.linalg.vector_norm(q, dim=-1)
    a_norm = torch.linalg.vector_norm(a, dim=-1)
    denom = torch.clamp(q_norm * a_norm, min=1.0e-12)
    cos_drift = 1.0 - (dot / denom)
    rel_l2 = torch.linalg.vector_norm(q - a, dim=-1) / torch.clamp(
        a_norm, min=1.0e-12
    )

    reduce = _aissd_q_drift_reduce()
    distribution: dict[str, Any] | None = None
    if collect_distribution:
        cos_stats = _aissd_q_drift_metric_stats(
            cos_drift, _aissd_q_drift_cos_threshold()
        )
        rel_stats = _aissd_q_drift_metric_stats(
            rel_l2, _aissd_q_drift_rel_l2_threshold()
        )
        max_cos = float(cos_stats.get("max", float("inf")))
        max_rel_l2 = float(rel_stats.get("max", float("inf")))
        if reduce == "p95":
            decision_cos = float(cos_stats.get("p95", float("inf")))
            decision_rel_l2 = float(rel_stats.get("p95", float("inf")))
        else:
            decision_cos = max_cos
            decision_rel_l2 = max_rel_l2
        distribution = {
            "active_reqs": int(cos_drift.shape[0]),
            "heads": int(cos_drift.shape[1]),
            "samples": int(cos_drift.numel()),
            "decision_reduce": reduce,
            "decision_cos": decision_cos,
            "decision_rel_l2": decision_rel_l2,
            "cos": cos_stats,
            "rel_l2": rel_stats,
        }
    else:
        max_cos = float(torch.amax(cos_drift).item())
        max_rel_l2 = float(torch.amax(rel_l2).item())
        if reduce == "p95":
            # This statistic is part of the live decision, so it must still be
            # computed when diagnostic logging is disabled. The tensors are tiny
            # (active requests x Q heads); only two scalar readbacks are needed.
            decision_cos = float(torch.quantile(cos_drift.reshape(-1), 0.95).item())
            decision_rel_l2 = float(torch.quantile(rel_l2.reshape(-1), 0.95).item())
        else:
            decision_cos = max_cos
            decision_rel_l2 = max_rel_l2

    values = (decision_cos, decision_rel_l2, max_cos, max_rel_l2)
    if any(not (value == value) for value in values):
        return float("inf"), float("inf"), float("inf"), float("inf"), distribution
    return decision_cos, decision_rel_l2, max_cos, max_rel_l2, distribution


def _aissd_q_drift_log_value(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.8f}"
    except Exception:
        return "NA"


def _aissd_log_q_drift_decision(
    *,
    step_context: dict[str, Any],
    layer_name: str,
    layer_id: int,
    token_state: dict[str, Any],
    restored: bool,
) -> None:
    # Reuse the existing token-reuse diagnostics switch; do not introduce a
    # second Q-drift-specific log switch. This emits only for F layers.
    if not _aissd_token_reuse_overhead_log_enabled():
        return

    cos_value = step_context.get("aissd_q_drift_cos_max")
    rel_value = step_context.get("aissd_q_drift_rel_l2_max")
    cos_decision = step_context.get("aissd_q_drift_cos_decision")
    rel_decision = step_context.get("aissd_q_drift_rel_l2_decision")
    decision_reduce = step_context.get(
        "aissd_q_drift_decision_reduce", _aissd_q_drift_reduce()
    )
    check_ms = float(step_context.get("aissd_q_drift_check_ms", 0.0) or 0.0)
    candidate_same = step_context.get("aissd_q_drift_candidate_same")
    anchor_step = step_context.get("aissd_q_drift_anchor_step")
    age = step_context.get("aissd_q_drift_anchor_age")
    reason = step_context.get("aissd_token_reuse_last_reason", "unknown")
    distribution = step_context.get("aissd_q_drift_distribution")

    if isinstance(distribution, dict):
        cos_stats = distribution.get("cos") or {}
        rel_stats = distribution.get("rel_l2") or {}
        metrics_state = "available"
        active_reqs: Any = distribution.get("active_reqs")
        heads: Any = distribution.get("heads")
        samples: Any = distribution.get("samples")
    else:
        cos_stats = {}
        rel_stats = {}
        metrics_state = "NA"
        active_reqs = "NA"
        heads = "NA"
        samples = "NA"

    logger.info(
        "[aissd-token-q-drift-threshold] layer=%s layer_id=%s token_step=%s "
        "anchor_step=%s anchor_age=%s candidate_same=%s decision_reduce=%s "
        "decision=%s reason=%s q_metrics=%s active_reqs=%s heads=%s samples=%s "
        "cos_threshold=%.8f cos_decision=%s cos_mean=%s cos_p50=%s cos_p90=%s cos_p95=%s "
        "cos_p99=%s cos_max=%s cos_exceed_ratio=%s cos_reqmax_mean=%s "
        "cos_reqmax_p50=%s cos_reqmax_p95=%s cos_reqmax_max=%s "
        "cos_req_exceed_ratio=%s cos_worst_req=%s cos_worst_head=%s "
        "rel_l2_threshold=%.8f rel_l2_decision=%s rel_l2_mean=%s rel_l2_p50=%s rel_l2_p90=%s "
        "rel_l2_p95=%s rel_l2_p99=%s rel_l2_max=%s rel_l2_exceed_ratio=%s "
        "rel_l2_reqmax_mean=%s rel_l2_reqmax_p50=%s rel_l2_reqmax_p95=%s "
        "rel_l2_reqmax_max=%s rel_l2_req_exceed_ratio=%s "
        "rel_l2_worst_req=%s rel_l2_worst_head=%s check_ms=%.6f",
        layer_name,
        layer_id,
        token_state.get("token_step"),
        anchor_step,
        age,
        candidate_same,
        decision_reduce,
        "reuse" if restored else "refresh",
        reason,
        metrics_state,
        active_reqs,
        heads,
        samples,
        _aissd_q_drift_cos_threshold(),
        _aissd_q_drift_log_value(cos_decision),
        _aissd_q_drift_log_value(cos_stats.get("mean")),
        _aissd_q_drift_log_value(cos_stats.get("p50")),
        _aissd_q_drift_log_value(cos_stats.get("p90")),
        _aissd_q_drift_log_value(cos_stats.get("p95")),
        _aissd_q_drift_log_value(cos_stats.get("p99")),
        _aissd_q_drift_log_value(cos_value),
        _aissd_q_drift_log_value(cos_stats.get("exceed_ratio")),
        _aissd_q_drift_log_value(cos_stats.get("reqmax_mean")),
        _aissd_q_drift_log_value(cos_stats.get("reqmax_p50")),
        _aissd_q_drift_log_value(cos_stats.get("reqmax_p95")),
        _aissd_q_drift_log_value(cos_stats.get("reqmax_max")),
        _aissd_q_drift_log_value(cos_stats.get("req_exceed_ratio")),
        cos_stats.get("worst_req", "NA"),
        cos_stats.get("worst_head", "NA"),
        _aissd_q_drift_rel_l2_threshold(),
        _aissd_q_drift_log_value(rel_decision),
        _aissd_q_drift_log_value(rel_stats.get("mean")),
        _aissd_q_drift_log_value(rel_stats.get("p50")),
        _aissd_q_drift_log_value(rel_stats.get("p90")),
        _aissd_q_drift_log_value(rel_stats.get("p95")),
        _aissd_q_drift_log_value(rel_stats.get("p99")),
        _aissd_q_drift_log_value(rel_value),
        _aissd_q_drift_log_value(rel_stats.get("exceed_ratio")),
        _aissd_q_drift_log_value(rel_stats.get("reqmax_mean")),
        _aissd_q_drift_log_value(rel_stats.get("reqmax_p50")),
        _aissd_q_drift_log_value(rel_stats.get("reqmax_p95")),
        _aissd_q_drift_log_value(rel_stats.get("reqmax_max")),
        _aissd_q_drift_log_value(rel_stats.get("req_exceed_ratio")),
        rel_stats.get("worst_req", "NA"),
        rel_stats.get("worst_head", "NA"),
        check_ms,
    )


def _aissd_token_reuse_virtual_step(
    step_context: dict[str, Any],
    generation: int,
    layer_id: int,
) -> int:
    """Return a Python-side decode-step counter for token reuse.

    req_token_lens is graph-stable in this path and can stay fixed across
    decode iterations, so it must not be used as the token-reuse clock.  The
    attention layers are visited in model order for each decode pass; when the
    layer id wraps back to an earlier layer, we advance the virtual token step.
    """
    gen_key = "aissd_token_reuse_virtual_generation"
    step_key = "aissd_token_reuse_virtual_step"
    last_layer_key = "aissd_token_reuse_virtual_last_layer_id"

    cur_generation = int(generation)
    cur_layer_id = int(layer_id)
    prev_generation = step_context.get(gen_key)
    if prev_generation is None or int(prev_generation) != cur_generation:
        step_context[gen_key] = cur_generation
        step_context[step_key] = 0
        step_context[last_layer_key] = cur_layer_id
        return 0

    step = int(step_context.get(step_key, 0) or 0)
    last_layer = step_context.get(last_layer_key)
    if last_layer is not None and cur_layer_id <= int(last_layer):
        step += 1
        step_context[step_key] = int(step)
    step_context[last_layer_key] = cur_layer_id
    return int(step)


def _aissd_token_reuse_state(
    step_context: dict[str, Any],
    generation: int,
    layer_id: int,
    strategy: str | None = None,
) -> dict[str, Any]:
    if _aissd_token_reuse_is_legacy(strategy):
        # Fast path for the old behavior: legacy reuse is gated only by
        # context_generation. Avoid reading req_token_lens/active_reqs because
        # converting tensors to Python lists can synchronize or add per-layer
        # CPU overhead on the decode path.
        return {
            "generation": int(generation),
            "token_step": int(generation),
            "active_sig": (),
            "token_sig": (),
            "legacy_fast_path": True,
        }

    req_ids = step_context.get("req_ids")
    if isinstance(req_ids, (list, tuple)):
        active_sig = tuple(str(x) for x in req_ids)
    else:
        active_reqs = _aissd_tensor_int_list(step_context.get("active_reqs"))
        if active_reqs:
            active_sig = tuple(str(x) for x in active_reqs)
        else:
            active_sig = (str(step_context.get("host_active_reqs", 0)),)

    token_step = _aissd_token_reuse_virtual_step(
        step_context,
        generation,
        layer_id,
    )
    token_sig = (int(token_step),)

    return {
        "generation": int(generation),
        "token_step": int(token_step),
        "active_sig": active_sig,
        "token_sig": token_sig,
    }


def _aissd_token_reuse_allows(
    entry: dict[str, Any],
    token_state: dict[str, Any],
    strategy: str | None = None,
    *,
    current_q: torch.Tensor | None = None,
    candidate_signature: tuple[int, ...] = (),
    evaluate_q_drift: bool = True,
    step_context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if strategy is None:
        strategy = _aissd_token_reuse_strategy()
    if _aissd_token_reuse_is_legacy(strategy):
        if int(entry.get("generation", -999999)) == int(token_state["generation"]):
            return True, "legacy_generation_match"
        return False, "legacy_generation_mismatch"

    if tuple(entry.get("active_sig", ())) != tuple(token_state.get("active_sig", ())):
        return False, "active_set_changed"

    entry_step = int(entry.get("token_step", -999999))
    cur_step = int(token_state["token_step"])
    delta = cur_step - entry_step
    if delta < 0:
        return False, "token_step_rewind"

    if _aissd_token_reuse_is_q_drift(strategy):
        # S layers never compare their Q with an F-layer anchor. They may consume
        # the source F layer only after that F layer resolved this token step.
        if not evaluate_q_drift:
            if int(entry.get("resolved_token_step", -999999)) == cur_step:
                return True, "q_drift_source_resolved"
            return False, "q_drift_source_not_resolved"

        if step_context is not None:
            step_context["aissd_q_drift_cos_max"] = None
            step_context["aissd_q_drift_rel_l2_max"] = None
            step_context["aissd_q_drift_cos_decision"] = None
            step_context["aissd_q_drift_rel_l2_decision"] = None
            step_context["aissd_q_drift_decision_reduce"] = _aissd_q_drift_reduce()
            step_context["aissd_q_drift_distribution"] = None
            step_context["aissd_q_drift_check_ms"] = 0.0
            step_context["aissd_q_drift_anchor_step"] = entry.get("anchor_token_step")
            step_context["aissd_q_drift_anchor_age"] = None
            step_context["aissd_q_drift_candidate_same"] = None

        anchor_step = int(entry.get("anchor_token_step", entry_step))
        age = cur_step - anchor_step
        if step_context is not None:
            step_context["aissd_q_drift_anchor_step"] = anchor_step
            step_context["aissd_q_drift_anchor_age"] = age
        if age < 0:
            return False, "q_drift_anchor_step_rewind"
        if age >= _aissd_q_drift_max_staleness():
            return False, "q_drift_max_staleness"

        cached_signature = tuple(entry.get("candidate_signature", ()))
        current_signature = tuple(candidate_signature)
        candidate_same = bool(cached_signature) and cached_signature == current_signature
        if step_context is not None:
            step_context["aissd_q_drift_candidate_same"] = candidate_same
        if not cached_signature:
            return False, "q_drift_candidate_signature_missing"
        if not current_signature:
            return False, "q_drift_current_candidate_signature_missing"
        if not candidate_same:
            return False, "q_drift_candidate_changed"

        anchor_q = entry.get("anchor_q")
        if not isinstance(anchor_q, torch.Tensor):
            return False, "q_drift_anchor_q_missing"
        if not isinstance(current_q, torch.Tensor):
            return False, "q_drift_current_q_missing"
        if tuple(anchor_q.shape) != tuple(current_q.shape):
            return False, "q_drift_q_shape_changed"

        drift_t0 = time.perf_counter()
        collect_distribution = _aissd_token_reuse_overhead_log_enabled()
        (
            cos_decision,
            rel_l2_decision,
            cos_max,
            rel_l2_max,
            distribution,
        ) = _aissd_compute_q_drift(
            current_q,
            anchor_q,
            collect_distribution=collect_distribution,
        )
        drift_ms = (time.perf_counter() - drift_t0) * 1000.0
        decision_reduce = _aissd_q_drift_reduce()
        if step_context is not None:
            step_context["aissd_q_drift_cos_max"] = float(cos_max)
            step_context["aissd_q_drift_rel_l2_max"] = float(rel_l2_max)
            step_context["aissd_q_drift_cos_decision"] = float(cos_decision)
            step_context["aissd_q_drift_rel_l2_decision"] = float(rel_l2_decision)
            step_context["aissd_q_drift_decision_reduce"] = decision_reduce
            step_context["aissd_q_drift_distribution"] = distribution
            step_context["aissd_q_drift_check_ms"] = float(drift_ms)
        if cos_decision > _aissd_q_drift_cos_threshold():
            return False, "q_drift_cos_threshold"
        if rel_l2_decision > _aissd_q_drift_rel_l2_threshold():
            return False, "q_drift_rel_l2_threshold"
        entry["resolved_token_step"] = cur_step
        entry["last_q_decision_reduce"] = decision_reduce
        entry["last_q_cos_drift"] = float(cos_decision)
        entry["last_q_rel_l2_drift"] = float(rel_l2_decision)
        entry["last_q_cos_max"] = float(cos_max)
        entry["last_q_rel_l2_max"] = float(rel_l2_max)
        return True, "q_drift_safe"

    if strategy in ("none", "off", "every_token", "per_token"):
        if delta == 0:
            return True, "same_token"
        return False, "every_token_refresh"

    if strategy in ("fixed_interval", "interval", "hybrid"):
        interval = _aissd_token_reuse_interval()
        max_staleness = _aissd_token_reuse_max_staleness()
        window = min(interval, max_staleness)
        if delta < window:
            return True, f"within_interval_{window}"
        return False, f"stale_delta_{delta}_ge_{window}"

    if strategy in ("always", "force"):
        return True, "force_reuse"

    if delta == 0:
        return True, f"unknown_strategy_{strategy}_same_token"
    return False, f"unknown_strategy_{strategy}_refresh"


def _sparse_kv_e2e_log_reuse_enabled() -> bool:
    # By default, sparse-KV e2e bandwidth is logged only on the layer that
    # really runs the AISSD selector.  Reuse layers have selector_ms=0 and would
    # otherwise produce misleading, inflated bandwidth numbers.
    return _env_flag("AISSD_SPARSE_KV_E2E_LOG_REUSE", "0")


def _aissd_backend_code(name: Any) -> int:
    value = str(name or "host").lower()
    if value == "ssd-cpu":
        return 1
    if value == "ssd-npu":
        return 2
    return 0


_AISSD_CANDIDATE_LAYER_KEYS = (
    ("aissd_candidate_count", "aissd_layer_candidate_count"),
    ("aissd_candidate_chunk_ids", "aissd_layer_candidate_chunk_ids"),
    ("aissd_candidate_block_ids", "aissd_layer_candidate_block_ids"),
    ("aissd_candidate_block_lens", "aissd_layer_candidate_block_lens"),
    ("aissd_candidate_token_start", "aissd_layer_candidate_token_start"),
    ("aissd_candidate_token_end", "aissd_layer_candidate_token_end"),
    ("aissd_candidate_dtype", "aissd_layer_candidate_dtype"),
    ("aissd_candidate_fmt", "aissd_layer_candidate_fmt"),
    ("aissd_candidate_ndim", "aissd_layer_candidate_ndim"),
    ("aissd_candidate_shape", "aissd_layer_candidate_shape"),
    ("aissd_candidate_extent_count", "aissd_layer_candidate_extent_count"),
    ("aissd_candidate_extent_lba", "aissd_layer_candidate_extent_lba"),
    ("aissd_candidate_extent_bytes", "aissd_layer_candidate_extent_bytes"),
    ("aissd_candidate_signature", "aissd_layer_candidate_signature"),
)


_AISSD_SELECTED_METADATA_KEYS = (
    "selected_block_table",
    "selected_block_lens",
    "selected_ready_flags",
    "fa_block_table",
    "fa_seq_lens",
)


def _aissd_candidate_layer_ids(step_context: dict[str, Any]) -> list[int]:
    ids = step_context.get("aissd_candidate_layer_ids")
    if isinstance(ids, torch.Tensor):
        try:
            return [int(x) for x in ids.detach().cpu().tolist()]
        except Exception:
            return []
    if isinstance(ids, (list, tuple)):
        try:
            return [int(x) for x in ids]
        except Exception:
            return []
    return []


def _aissd_select_candidate_tensors_for_layer(
    step_context: dict[str, Any],
    layer_id: int,
) -> None:
    layer_ids = _aissd_candidate_layer_ids(step_context)
    if not layer_ids:
        return
    if int(layer_id) not in layer_ids:
        raise RuntimeError(
            "AISSD static selector requested layer_id="
            f"{layer_id}, but prepare_sparse_kv_step() only built candidate "
            f"metadata for layers={layer_ids}. Check AISSD_F_LAYERS and "
            "qkpack layer configuration."
        )
    layer_pos = layer_ids.index(int(layer_id))
    for target_key, layered_key in _AISSD_CANDIDATE_LAYER_KEYS:
        layered = step_context.get(layered_key)
        if not isinstance(layered, torch.Tensor):
            continue
        if layered.dim() <= 0 or layer_pos >= int(layered.shape[0]):
            raise RuntimeError(
                f"AISSD layered candidate tensor {layered_key} has invalid "
                f"shape={tuple(layered.shape)} for layer_pos={layer_pos}"
            )
        step_context[target_key] = layered[layer_pos]
    step_context["aissd_candidate_active_layer_id"] = int(layer_id)


def _aissd_selected_cache(step_context: dict[str, Any]) -> dict[int, dict[str, Any]]:
    cache = step_context.get("aissd_selected_metadata_cache")
    if not isinstance(cache, dict):
        cache = {}
        step_context["aissd_selected_metadata_cache"] = cache
    return cache


def _aissd_cache_selected_metadata(
    step_context: dict[str, Any],
    source_layer_id: int,
    source_layer_name: str,
    generation: int,
    token_state: dict[str, Any],
    strategy: str | None = None,
    *,
    current_q: torch.Tensor | None = None,
    candidate_signature: tuple[int, ...] = (),
) -> None:
    if strategy is None:
        strategy = _aissd_token_reuse_strategy()
    entry: dict[str, Any] = {
        "generation": int(generation),
        "source_layer_id": int(source_layer_id),
        "source_layer_name": str(source_layer_name),
        "token_step": int(token_state["token_step"]),
        "resolved_token_step": int(token_state["token_step"]),
        "active_sig": tuple(token_state.get("active_sig", ())),
        "token_sig": tuple(token_state.get("token_sig", ())),
        "token_reuse_strategy": strategy,
    }
    if _aissd_token_reuse_is_q_drift(strategy):
        if not isinstance(current_q, torch.Tensor):
            raise RuntimeError(
                "AISSD q_drift selector refresh cannot cache metadata without "
                f"the current F-layer Q, layer={source_layer_name}"
            )
        if not candidate_signature:
            raise RuntimeError(
                "AISSD q_drift selector refresh cannot cache metadata without "
                f"a candidate signature, layer={source_layer_name}"
            )
        entry["anchor_q"] = current_q.detach().clone()
        entry["anchor_token_step"] = int(token_state["token_step"])
        entry["candidate_signature"] = tuple(candidate_signature)
        step_context["aissd_q_drift_anchor_copy_bytes"] = (
            int(current_q.numel()) * int(current_q.element_size())
        )
    else:
        step_context["aissd_q_drift_anchor_copy_bytes"] = 0

    copy_bytes = 0
    for key in _AISSD_SELECTED_METADATA_KEYS:
        tensor = step_context.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"AISSD selected metadata cache missing tensor {key}")
        entry[key] = tensor.detach().clone()
        copy_bytes += int(tensor.numel()) * int(tensor.element_size())
    _aissd_selected_cache(step_context)[int(source_layer_id)] = entry
    step_context["aissd_selected_metadata_active_layer_id"] = int(source_layer_id)
    step_context["aissd_selected_metadata_active_layer_name"] = str(source_layer_name)
    step_context["aissd_selected_metadata_active_generation"] = int(generation)
    step_context["aissd_selected_metadata_active_token_step"] = int(token_state["token_step"])
    step_context["aissd_token_reuse_cache_copy_count"] = len(_AISSD_SELECTED_METADATA_KEYS)
    step_context["aissd_token_reuse_cache_copy_bytes"] = int(copy_bytes)


def _aissd_selected_metadata_is_active(
    step_context: dict[str, Any],
    source_layer_id: int,
    generation: int,
    token_state: dict[str, Any],
    strategy: str,
) -> bool:
    if int(step_context.get("aissd_selected_metadata_active_layer_id", -999999)) != int(
        source_layer_id
    ):
        return False
    if int(step_context.get("aissd_selected_metadata_active_generation", -999999)) != int(
        generation
    ):
        return False
    if _aissd_token_reuse_is_legacy(strategy):
        return True
    return int(step_context.get("aissd_selected_metadata_active_token_step", -999999)) == int(
        token_state["token_step"]
    )


def _aissd_restore_selected_metadata(
    step_context: dict[str, Any],
    source_layer_id: int,
    generation: int,
    token_state: dict[str, Any],
    strategy: str | None = None,
    *,
    current_q: torch.Tensor | None = None,
    candidate_signature: tuple[int, ...] = (),
    evaluate_q_drift: bool = True,
) -> bool:
    if strategy is None:
        strategy = _aissd_token_reuse_strategy()
    step_context["aissd_token_reuse_restore_copy_count"] = 0
    step_context["aissd_token_reuse_restore_copy_bytes"] = 0
    step_context["aissd_token_reuse_restore_skip_active"] = False
    entry = _aissd_selected_cache(step_context).get(int(source_layer_id))
    if not isinstance(entry, dict):
        step_context["aissd_token_reuse_last_reason"] = "missing_cache_entry"
        if _aissd_token_reuse_is_q_drift(strategy):
            step_context["aissd_q_drift_cos_max"] = None
            step_context["aissd_q_drift_rel_l2_max"] = None
            step_context["aissd_q_drift_distribution"] = None
            step_context["aissd_q_drift_check_ms"] = 0.0
            step_context["aissd_q_drift_anchor_step"] = None
            step_context["aissd_q_drift_anchor_age"] = None
            step_context["aissd_q_drift_candidate_same"] = None
        return False
    allowed, reason = _aissd_token_reuse_allows(
        entry,
        token_state,
        strategy,
        current_q=current_q,
        candidate_signature=candidate_signature,
        evaluate_q_drift=evaluate_q_drift,
        step_context=step_context,
    )
    step_context["aissd_token_reuse_last_reason"] = reason
    step_context["aissd_token_reuse_cached_token_step"] = entry.get("token_step")
    if not allowed:
        return False

    if _aissd_selected_metadata_is_active(
        step_context, source_layer_id, generation, token_state, strategy
    ):
        step_context["aissd_token_reuse_restore_skip_active"] = True
        step_context["aissd_token_reuse_last_reason"] = f"{reason}_already_active"
        return True

    copy_count = 0
    copy_bytes = 0
    for key in _AISSD_SELECTED_METADATA_KEYS:
        cached = entry.get(key)
        dst = step_context.get(key)
        if not isinstance(cached, torch.Tensor) or not isinstance(dst, torch.Tensor):
            return False
        if tuple(cached.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"AISSD selected metadata cache shape mismatch for {key}: "
                f"cached={tuple(cached.shape)} dst={tuple(dst.shape)}"
            )
        dst.copy_(cached, non_blocking=True)
        copy_count += 1
        copy_bytes += int(cached.numel()) * int(cached.element_size())
    step_context["aissd_selected_metadata_active_layer_id"] = int(source_layer_id)
    step_context["aissd_selected_metadata_active_layer_name"] = str(
        entry.get("source_layer_name", source_layer_id)
    )
    step_context["aissd_selected_metadata_active_generation"] = int(generation)
    step_context["aissd_selected_metadata_active_token_step"] = int(token_state["token_step"])
    step_context["aissd_token_reuse_last_reason"] = reason
    step_context["aissd_token_reuse_restore_copy_count"] = int(copy_count)
    step_context["aissd_token_reuse_restore_copy_bytes"] = int(copy_bytes)
    return True


def _maybe_run_aissd_selector_op(
    query: torch.Tensor,
    step_context: dict[str, Any],
    layer_name: str,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
) -> float:
    """Run AISSD q-aware selector before sparse FlashAttention.

    This is not the old Attention.forward Python hook.  The C++ op is invoked
    from the sparse attention backend and consumes the real Q tensor plus the
    native LMCache candidate extent metadata prepared in the step context.
    The C++ implementation is registered from sparse_flash_attention.cu, so it
    reuses the existing sparse attention custom-op build path.
    """
    backend_name = step_context.get("aissd_selector_backend", "host")
    backend = _aissd_backend_code(backend_name)
    if backend == 0:
        step_context["aissd_selector_ms"] = 0.0
        step_context["aissd_selector_reused"] = False
        step_context["aissd_selector_real_layer"] = None
        return 0.0

    # CUDA graph capture / warmup can reach this backend before a real
    # SchedulerOutput has published candidate native extents.  Do not run AISSD
    # RPC here: the selector is meaningful only for real steps with active
    # requests and candidate extents.  This is not a runtime fallback; it is a
    # bootstrap/capture guard.
    try:
        host_active_reqs = int(step_context.get("host_active_reqs", 0) or 0)
    except Exception:
        host_active_reqs = 0
    if host_active_reqs <= 0:
        if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
            logger.info(
                "[aissd-selector-op] skip bootstrap/capture layer=%s backend=%s "
                "host_active_reqs=%s",
                layer_name,
                backend_name,
                step_context.get("host_active_reqs"),
            )
        step_context["aissd_selector_ms"] = 0.0
        step_context["aissd_selector_reused"] = False
        step_context["aissd_selector_real_layer"] = None
        return 0.0
    if torch.cuda.is_current_stream_capturing():
        # This should normally only happen during CUDA graph capture with dummy
        # inputs.  Host RPC/CMB/SSD IO is not CUDA-graph-capturable, so never
        # issue AISSD RPC while a stream capture is active.
        if _sparse_kv_debug_enabled():
            logger.info(
                "[aissd-selector-op] skip CUDA graph capture layer=%s backend=%s",
                layer_name,
                backend_name,
            )
        step_context["aissd_selector_ms"] = 0.0
        step_context["aissd_selector_reused"] = False
        step_context["aissd_selector_real_layer"] = None
        return 0.0

    generation = int(step_context.get("context_generation", -1) or -1)
    token_reuse_strategy = _aissd_token_reuse_strategy()
    overhead_log_enabled = _aissd_token_reuse_overhead_log_enabled()
    layer_id = _aissd_parse_layer_id(layer_name, step_context)
    layer_reuse_enabled = _aissd_layer_reuse_enabled()
    layer_reuse_strategy = _aissd_layer_reuse_strategy()
    if _aissd_token_reuse_is_q_drift(token_reuse_strategy) and (
        not layer_reuse_enabled or layer_reuse_strategy != "static"
    ):
        raise RuntimeError(
            "AISSD_TOKEN_REUSE_STRATEGY=q_drift currently requires "
            "AISSD_SPARSE_KV_LAYER_REUSE=1 and "
            "AISSD_LAYER_REUSE_STRATEGY=static"
        )

    token_state_t0 = time.perf_counter()
    token_state = _aissd_token_reuse_state(
        step_context, generation, layer_id, token_reuse_strategy
    )
    token_state_ms = (time.perf_counter() - token_state_t0) * 1000.0
    q_drift_query: torch.Tensor | None = None
    candidate_signature: tuple[int, ...] = ()

    if layer_reuse_enabled:
        done_generation = int(
            step_context.get("aissd_selector_done_generation", -999999) or -999999
        )
        done_layer_id = int(
            step_context.get("aissd_selector_done_layer_id", -999999) or -999999
        )
        strategy = layer_reuse_strategy
        if strategy == "static":
            f_layers = _aissd_static_f_layers()
            is_f_layer = int(layer_id) in f_layers
            source_layer_id = _aissd_static_reuse_source_layer_id(layer_id, f_layers)

            if _aissd_token_reuse_is_q_drift(token_reuse_strategy) and is_f_layer:
                # Activate this F layer's candidate metadata before comparing its
                # signature with the signature cached at the previous selector.
                _aissd_select_candidate_tensors_for_layer(step_context, layer_id)
                candidate_signature = _aissd_current_candidate_signature(step_context)
                q_drift_query = _aissd_q_drift_query_view(
                    query,
                    host_active_reqs,
                    int(num_heads),
                    int(head_size),
                )
                if q_drift_query is None:
                    raise RuntimeError(
                        "AISSD q_drift could not reshape the current F-layer Q to "
                        f"[active_reqs, num_heads, head_size]: layer={layer_name} "
                        f"q_shape={tuple(query.shape)} active_reqs={host_active_reqs} "
                        f"num_heads={num_heads} head_size={head_size}"
                    )
                if not candidate_signature:
                    raise RuntimeError(
                        "AISSD q_drift requires candidate signatures from "
                        "prepare_sparse_kv_step(); update vllm_v1_adapter.py and "
                        f"verify layer={layer_name}"
                    )

            restore_t0 = time.perf_counter()
            restored = _aissd_restore_selected_metadata(
                step_context,
                source_layer_id,
                generation,
                token_state,
                token_reuse_strategy,
                current_q=q_drift_query,
                candidate_signature=candidate_signature,
                evaluate_q_drift=(
                    _aissd_token_reuse_is_q_drift(token_reuse_strategy)
                    and is_f_layer
                ),
            )
            restore_ms = (time.perf_counter() - restore_t0) * 1000.0
            if (
                _aissd_token_reuse_is_q_drift(token_reuse_strategy)
                and is_f_layer
            ):
                _aissd_log_q_drift_decision(
                    step_context=step_context,
                    layer_name=layer_name,
                    layer_id=layer_id,
                    token_state=token_state,
                    restored=restored,
                )
            if overhead_log_enabled:
                logger.info(
                    "[aissd-token-reuse-overhead] op=restore layer=%s layer_id=%s "
                    "generation=%s strategy=%s source_layer_id=%s hit=%s "
                    "state_ms=%.6f restore_ms=%.6f copy_count=%s copy_bytes=%s "
                    "skip_active=%s token_step=%s cached_token_step=%s reason=%s "
                    "q_drift_check_ms=%s",
                    layer_name,
                    layer_id,
                    generation,
                    token_reuse_strategy,
                    source_layer_id,
                    restored,
                    token_state_ms,
                    restore_ms,
                    step_context.get("aissd_token_reuse_restore_copy_count", 0),
                    step_context.get("aissd_token_reuse_restore_copy_bytes", 0),
                    step_context.get("aissd_token_reuse_restore_skip_active", False),
                    token_state.get("token_step"),
                    step_context.get("aissd_token_reuse_cached_token_step"),
                    step_context.get("aissd_token_reuse_last_reason"),
                    step_context.get("aissd_q_drift_check_ms", 0.0),
                )
            if restored:
                if _aissd_selector_stats_enabled() or _sparse_kv_debug_enabled():
                    logger.info(
                        "[aissd-selector-op] reuse layer=%s layer_id=%s "
                        "generation=%s strategy=static source_layer_id=%s "
                        "source_layer_id_active=%s token_reuse_strategy=%s "
                        "token_step=%s cached_token_step=%s token_reuse_reason=%s "
                        "f_layers=%s",
                        layer_name,
                        layer_id,
                        generation,
                        source_layer_id,
                        step_context.get("aissd_selected_metadata_active_layer_id"),
                        token_reuse_strategy,
                        token_state.get("token_step"),
                        step_context.get("aissd_token_reuse_cached_token_step"),
                        step_context.get("aissd_token_reuse_last_reason"),
                        ",".join(str(x) for x in f_layers),
                    )
                step_context["aissd_selector_ms"] = 0.0
                step_context["aissd_selector_reused"] = True
                step_context["aissd_selector_reuse_strategy"] = "static"
                step_context["aissd_selector_token_reuse_strategy"] = token_reuse_strategy
                step_context["aissd_selector_token_step"] = int(token_state["token_step"])
                step_context["aissd_selector_token_reuse_reason"] = step_context.get(
                    "aissd_token_reuse_last_reason"
                )
                step_context["aissd_selector_reuse_source_layer_id"] = source_layer_id
                step_context["aissd_selector_real_layer"] = step_context.get(
                    "aissd_selected_metadata_active_layer_name"
                )
                step_context["aissd_selector_real_layer_id"] = source_layer_id
                return 0.0
            if not is_f_layer:
                raise RuntimeError(
                    "AISSD static reuse metadata is not ready for non-F layer "
                    f"layer={layer_name} layer_id={layer_id} source_layer_id={source_layer_id} "
                    f"generation={generation}. This layer must reuse selected metadata "
                    "from its nearest previous F layer; it must not fallback to running "
                    "a selector."
                )
            if not _aissd_token_reuse_is_q_drift(token_reuse_strategy):
                _aissd_select_candidate_tensors_for_layer(step_context, layer_id)
            if _aissd_selector_stats_enabled() or _sparse_kv_debug_enabled():
                logger.info(
                    "[aissd-selector-op] run layer=%s layer_id=%s generation=%s "
                    "strategy=static candidate_layer_id=%s token_reuse_strategy=%s "
                    "token_step=%s previous_cached_token_step=%s token_reuse_reason=%s "
                    "f_layers=%s",
                    layer_name,
                    layer_id,
                    generation,
                    step_context.get("aissd_candidate_active_layer_id"),
                    token_reuse_strategy,
                    token_state.get("token_step"),
                    step_context.get("aissd_token_reuse_cached_token_step"),
                    step_context.get("aissd_token_reuse_last_reason"),
                    ",".join(str(x) for x in f_layers),
                )
        elif done_generation == generation:
            if _aissd_selector_stats_enabled() or _sparse_kv_debug_enabled():
                logger.info(
                    "[aissd-selector-op] reuse layer=%s generation=%s "
                    "strategy=%s first_layer=%s",
                    layer_name,
                    generation,
                    strategy or "global",
                    step_context.get("aissd_selector_done_layer"),
                )
            step_context["aissd_selector_ms"] = 0.0
            step_context["aissd_selector_reused"] = True
            step_context["aissd_selector_reuse_strategy"] = strategy or "global"
            step_context["aissd_selector_real_layer"] = step_context.get("aissd_selector_done_layer")
            step_context["aissd_selector_real_layer_id"] = step_context.get("aissd_selector_done_layer_id")
            return 0.0
    else:
        # AISSD_SPARSE_KV_LAYER_REUSE=0 is the hard off switch for layer reuse.
        # In this mode each attention layer must run its own selector using that
        # layer's prepared candidate/qkpack metadata.  Do not fall back to the
        # legacy non-layered aliases, because those aliases point at candidate
        # layer position 0 in the layered step context.
        _aissd_select_candidate_tensors_for_layer(step_context, layer_id)
        step_context["aissd_selector_reuse_strategy"] = "none"
        step_context["aissd_selector_reuse_source_layer_id"] = int(layer_id)
        if _aissd_selector_stats_enabled() or _sparse_kv_debug_enabled():
            logger.info(
                "[aissd-selector-op] run layer=%s layer_id=%s generation=%s "
                "strategy=none candidate_layer_id=%s token_reuse_strategy=%s "
                "token_step=%s",
                layer_name,
                layer_id,
                generation,
                step_context.get("aissd_candidate_active_layer_id"),
                token_reuse_strategy,
                token_state.get("token_step"),
            )

    required = (
        "aissd_candidate_count",
        "aissd_candidate_chunk_ids",
        "aissd_candidate_block_ids",
        "aissd_candidate_block_lens",
        "aissd_candidate_token_start",
        "aissd_candidate_token_end",
        "aissd_candidate_dtype",
        "aissd_candidate_fmt",
        "aissd_candidate_ndim",
        "aissd_candidate_shape",
        "aissd_candidate_extent_count",
        "aissd_candidate_extent_lba",
        "aissd_candidate_extent_bytes",
        "fa_block_table",
        "fa_seq_lens",
    )
    missing = [k for k in required if k not in step_context]
    if missing:
        raise RuntimeError(
            f"AISSD selector backend={backend_name} requested for layer={layer_name}, "
            f"and host_active_reqs={host_active_reqs}, but sparse step context "
            f"lacks native extent tensors: {missing}. This indicates "
            "prepare_sparse_kv_step() did not publish AISSD candidate extents "
            "for a real request."
        )
    if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
        logger.info(
            "[aissd-selector-op] layer=%s backend=%s q_shape=%s active_reqs=%s",
            layer_name,
            backend_name,
            tuple(query.shape),
            step_context.get("host_active_reqs"),
        )
    t0 = time.perf_counter()
    selector_prof = _sparse_profile_begin("aissd_selector_op", layer_name, query)
    vllm_ops.aissd_sparse_kv_select(
        query,
        step_context["active_reqs"],
        step_context["req_token_lens"],
        step_context["req_lmcache_cached_tokens"],
        step_context["aissd_candidate_count"],
        step_context["aissd_candidate_chunk_ids"],
        step_context["aissd_candidate_block_ids"],
        step_context["aissd_candidate_block_lens"],
        step_context["aissd_candidate_token_start"],
        step_context["aissd_candidate_token_end"],
        step_context["aissd_candidate_dtype"],
        step_context["aissd_candidate_fmt"],
        step_context["aissd_candidate_ndim"],
        step_context["aissd_candidate_shape"],
        step_context["aissd_candidate_extent_count"],
        step_context["aissd_candidate_extent_lba"],
        step_context["aissd_candidate_extent_bytes"],
        step_context["selected_block_table"],
        step_context["selected_block_lens"],
        step_context["selected_ready_flags"],
        step_context["fa_block_table"],
        step_context["fa_seq_lens"],
        layer_id,
        backend,
        int(num_heads),
        int(num_kv_heads),
        int(head_size),
        int(step_context["chunk_size"]),
        int(step_context["block_size"]),
        int(step_context["top_n_chunks"]),
        int(step_context.get("aissd_top_m", 8)),
        int(step_context.get("aissd_score_mode_code", 1)),
        int(step_context.get("aissd_manifest_block_size", 4096)),
        int(step_context.get("aissd_timeout_ms", 300000)),
    )
    if layer_reuse_enabled and layer_reuse_strategy == "static":
        cache_t0 = time.perf_counter()
        _aissd_cache_selected_metadata(
            step_context,
            layer_id,
            layer_name,
            generation,
            token_state,
            token_reuse_strategy,
            current_q=q_drift_query,
            candidate_signature=candidate_signature,
        )
        cache_ms = (time.perf_counter() - cache_t0) * 1000.0
        if overhead_log_enabled:
            logger.info(
                "[aissd-token-reuse-overhead] op=cache layer=%s layer_id=%s "
                "generation=%s strategy=%s state_ms=%.6f cache_ms=%.6f "
                "copy_count=%s copy_bytes=%s token_step=%s",
                layer_name,
                layer_id,
                generation,
                token_reuse_strategy,
                token_state_ms,
                cache_ms,
                step_context.get("aissd_token_reuse_cache_copy_count", 0),
                step_context.get("aissd_token_reuse_cache_copy_bytes", 0),
                token_state.get("token_step"),
            )
    _sparse_profile_end(selector_prof, step_context, query, impl=str(backend_name))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    step_context["aissd_selector_ms"] = float(elapsed_ms)
    step_context["aissd_selector_reused"] = False
    step_context["aissd_selector_reuse_strategy"] = (
        layer_reuse_strategy if layer_reuse_enabled else "none"
    )
    step_context["aissd_selector_reuse_source_layer_id"] = int(layer_id)
    step_context["aissd_selector_token_reuse_strategy"] = token_reuse_strategy
    step_context["aissd_selector_token_step"] = int(token_state["token_step"])
    step_context["aissd_selector_token_reuse_reason"] = "selector_refreshed"
    if layer_reuse_enabled and layer_reuse_strategy == "static":
        step_context["aissd_selector_f_layers"] = _aissd_static_f_layers()
    else:
        step_context["aissd_selector_f_layers"] = ()
    step_context["aissd_selector_real_layer"] = str(layer_name)
    step_context["aissd_selector_real_layer_id"] = int(layer_id)
    step_context["aissd_selector_real_generation"] = int(generation)
    step_context["aissd_selector_last_layer"] = str(layer_name)
    step_context["aissd_selector_last_layer_id"] = int(layer_id)
    step_context["aissd_selector_last_generation"] = int(generation)
    step_context["aissd_selector_done_generation"] = generation
    step_context["aissd_selector_done_layer"] = str(layer_name)
    step_context["aissd_selector_done_layer_id"] = int(layer_id)
    if _aissd_selector_stats_enabled():
        logger.info(
            "[aissd-selector-latency] layer=%s generation=%s backend=%s "
            "elapsed_ms=%.3f active_reqs=%s candidates=%s top_n=%s "
            "reuse_layers=%s reuse_strategy=%s token_reuse_strategy=%s "
            "token_step=%s f_layers=%s",
            layer_name,
            generation,
            backend_name,
            elapsed_ms,
            step_context.get("host_active_reqs"),
            step_context.get("aissd_candidate_count"),
            step_context.get("top_n_chunks"),
            layer_reuse_enabled,
            step_context.get("aissd_selector_reuse_strategy"),
            step_context.get("aissd_selector_token_reuse_strategy"),
            step_context.get("aissd_selector_token_step"),
            ",".join(str(x) for x in _aissd_static_f_layers())
            if layer_reuse_enabled and layer_reuse_strategy == "static"
            else "",
        )


    return float(elapsed_ms)

def _ensure_sparse_debug_counters(
    step_context: dict[str, Any],
    device: torch.device,
    enabled: bool,
) -> torch.Tensor:
    # Pass a stable tensor to the C++ op.  When debug is disabled, use a
    # zero-length tensor so the CUDA kernel skips all debug atomics.
    key = "debug_counters" if enabled else "debug_counters_disabled"
    counters = step_context.get(key)
    expected_numel = 8 if enabled else 0
    if (
        not isinstance(counters, torch.Tensor)
        or counters.device != device
        or counters.dtype != torch.long
        or counters.numel() != expected_numel
    ):
        counters = torch.zeros(expected_numel, dtype=torch.long, device=device)
        step_context[key] = counters
    return counters


def _read_sparse_debug_counters(counters: torch.Tensor) -> list[int]:
    if not isinstance(counters, torch.Tensor) or counters.numel() < 8:
        return [0] * 8
    return [int(x) for x in counters.detach().cpu().tolist()[:8]]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            # Avoid synchronizing CUDA tensors for log-only metadata.
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, torch.Tensor):
            return default
        return int(value)
    except Exception:
        return default


def _sparse_gbps(num_bytes: int, ms: float) -> float:
    if num_bytes <= 0 or ms <= 0.0:
        return 0.0
    return float(num_bytes) / (float(ms) / 1000.0) / 1.0e9


def _log_sparse_kv_e2e_bandwidth(
    *,
    step_context: dict[str, Any] | None,
    layer_name: str,
    impl: str,
    q_shape: Any,
    selector_ms: float,
    attention_ms: float,
) -> None:
    """Log sparse-KV end-to-end bandwidth from HOST selector start to FA end.

    The numerator is selected KV bytes.  When real selected-load instrumentation
    is available from LMCache, use it; otherwise use the step-level estimate
    prepared by the connector from selected blocks/chunks and model KV shape.
    """
    if step_context is None or not _aissd_sparse_kv_e2e_stats_enabled(step_context):
        return

    selector_reused = bool(step_context.get("aissd_selector_reused", False))
    real_layer = step_context.get("aissd_selector_real_layer")
    if (
        selector_reused
        and not _sparse_kv_e2e_log_reuse_enabled()
        and str(layer_name) != str(real_layer)
    ):
        if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
            logger.info(
                "[sparse-kv-e2e-bandwidth] skip_reuse layer=%s generation=%s "
                "real_layer=%s selector_ms=%.3f",
                layer_name,
                step_context.get("context_generation"),
                real_layer,
                float(selector_ms),
            )
        return

    if selector_ms <= 0.0 and not _sparse_kv_e2e_log_reuse_enabled():
        # Do not report end-to-end bandwidth without a real selector timing.
        # This happens on bootstrap/capture or layer-reuse paths.
        return

    load_ms = _as_float(step_context.get("sparse_selected_load_ms"), 0.0)
    load_wall_ms = _as_float(step_context.get("sparse_selected_load_wall_ms"), load_ms)
    load_bytes = _as_int(step_context.get("sparse_selected_load_bytes"), 0)
    host_prepare_ms = _as_float(step_context.get("sparse_host_prepare_ms"), 0.0)
    candidate_build_ms = _as_float(step_context.get("sparse_candidate_build_ms"), 0.0)
    sparse_attn_prepare_ms = _as_float(step_context.get("sparse_attn_prepare_ms"), 0.0)
    sparse_attention_kernel_ms = _as_float(step_context.get("sparse_attention_kernel_ms"), max(0.0, float(attention_ms) - sparse_attn_prepare_ms))
    selected_bytes = _as_int(step_context.get("sparse_selected_kv_bytes"), 0)
    if selected_bytes <= 0:
        selected_bytes = load_bytes
    if load_bytes <= 0:
        load_bytes = selected_bytes

    selected_tokens = _as_int(step_context.get("sparse_selected_tokens"), 0)
    selected_blocks = _as_int(step_context.get("sparse_selected_blocks"), 0)
    host_reqs = _as_int(step_context.get("host_active_reqs"), 0)
    candidates = step_context.get("aissd_candidate_count")
    if isinstance(candidates, torch.Tensor):
        candidates_repr = f"Tensor(shape={tuple(candidates.shape)}, device={candidates.device})"
    else:
        candidates_repr = candidates

    # Selector -> selected KV ready -> sparse-attention input metadata prepared.
    # sparse_kv_e2e_no_attn follows the experiment definition and stops before
    # the attention kernel itself.  The legacy attention_ms is also logged.
    selector_wall_ms = float(selector_ms)
    selected_load_wall_ms = float(load_wall_ms)
    e2e_no_attn_ms = selector_wall_ms + selected_load_wall_ms + float(sparse_attn_prepare_ms)
    e2e_with_attn_ms = e2e_no_attn_ms + float(sparse_attention_kernel_ms)
    logger.info(
        "[sparse-kv-e2e-bandwidth] layer=%s generation=%s impl=%s "
        "q_shape=%s active_reqs=%s candidates=%s selected_blocks=%d "
        "selected_tokens=%d selected_kv_bytes=%d selector_ms=%.3f "
        "selected_load_ms=%.3f attention_ms=%.3f e2e_no_attn_ms=%.3f "
        "e2e_with_attn_ms=%.3f selected_load_bw_GBps=%.6f "
        "e2e_no_attn_bw_GBps=%.6f e2e_with_attn_bw_GBps=%.6f "
        "bytes_source=%s selector_reused=%s real_selector_layer=%s",
        layer_name,
        step_context.get("context_generation"),
        impl,
        q_shape,
        host_reqs,
        candidates_repr,
        selected_blocks,
        selected_tokens,
        selected_bytes,
        float(selector_ms),
        float(load_ms),
        float(attention_ms),
        e2e_no_attn_ms,
        e2e_with_attn_ms,
        _sparse_gbps(load_bytes, load_ms),
        _sparse_gbps(selected_bytes, e2e_no_attn_ms),
        _sparse_gbps(selected_bytes, e2e_with_attn_ms),
        step_context.get("sparse_selected_kv_bytes_source", "unknown"),
        selector_reused,
        real_layer,
    )

    logger.info(
        "[aissd-sparse-kv-e2e-breakdown] layer=%s generation=%s impl=%s "
        "active_reqs=%d candidate_counts=%s selected_blocks=%d selected_tokens=%d "
        "selected_kv_bytes=%d host_prepare_ms=%.3f candidate_build_ms=%.3f "
        "selector_wall_ms=%.3f selected_kv_load_sum_ms=%.3f "
        "selected_kv_load_wall_ms=%.3f sparse_attn_prepare_ms=%.3f "
        "sparse_attn_kernel_ms=%.3f attention_total_ms=%.3f "
        "e2e_no_attn_ms=%.3f e2e_with_attn_ms=%.3f "
        "e2e_no_attn_bw_GBps=%.6f e2e_with_attn_bw_GBps=%.6f",
        layer_name,
        step_context.get("context_generation"),
        impl,
        host_reqs,
        step_context.get("sparse_candidate_counts", candidates_repr),
        selected_blocks,
        selected_tokens,
        selected_bytes,
        host_prepare_ms,
        candidate_build_ms,
        selector_wall_ms,
        float(load_ms),
        selected_load_wall_ms,
        float(sparse_attn_prepare_ms),
        float(sparse_attention_kernel_ms),
        float(attention_ms),
        e2e_no_attn_ms,
        e2e_with_attn_ms,
        _sparse_gbps(selected_bytes, e2e_no_attn_ms),
        _sparse_gbps(selected_bytes, e2e_with_attn_ms),
    )


def _sparse_tensor_ptr(tensor: Any) -> str:
    if isinstance(tensor, torch.Tensor):
        try:
            return hex(int(tensor.data_ptr()))
        except Exception:
            return "unavailable"
    return "None"


def _log_sparse_context_ptr(tag: str, layer_name: str, context: dict[str, Any] | None) -> None:
    if not _sparse_ctx_ptr_debug_enabled():
        return
    if context is None:
        logger.info("[sparse-ctx-ptr][%s] layer=%s ctx=None", tag, layer_name)
        return
    logger.info(
        "[sparse-ctx-ptr][%s] layer=%s ctx_id=%s generation=%s host_reqs=%s "
        "host_selected_blocks=%s active_reqs_ptr=%s req_token_lens_ptr=%s "
        "slot_mapping_table_ptr=%s selected_block_table_ptr=%s "
        "selected_block_lens_ptr=%s selected_ready_flags_ptr=%s debug_counters_ptr=%s",
        tag,
        layer_name,
        hex(id(context)),
        context.get("context_generation"),
        context.get("host_active_reqs"),
        context.get("host_selected_blocks"),
        _sparse_tensor_ptr(context.get("active_reqs")),
        _sparse_tensor_ptr(context.get("req_token_lens")),
        _sparse_tensor_ptr(context.get("slot_mapping_table")),
        _sparse_tensor_ptr(context.get("selected_block_table")),
        _sparse_tensor_ptr(context.get("selected_block_lens")),
        _sparse_tensor_ptr(context.get("selected_ready_flags")),
        _sparse_tensor_ptr(context.get("debug_counters")),
    )


def _get_sparse_connector():
    try:
        from lmcache.integration.vllm.vllm_v1_adapter import (
            get_sparse_kv_connector,
        )

        return get_sparse_kv_connector()
    except Exception:
        return None


def _ctx_generation(context: dict[str, Any] | None) -> int:
    if context is None:
        return -1
    try:
        return int(context.get("context_generation", -1) or -1)
    except Exception:
        return -1


def _ctx_host_active_reqs(context: dict[str, Any] | None) -> int:
    if context is None:
        return 0
    try:
        return int(context.get("host_active_reqs", 0) or 0)
    except Exception:
        return 0


def _get_sparse_step_context(attn_metadata: AttentionMetadata) -> dict[str, Any] | None:
    # There are two possible sources:
    #   1. attn_metadata/common_metadata.sparse_kv_step_context
    #   2. the LMCache connector's current persistent context
    # In CUDA graph / compile paths, attn_metadata may keep an older empty
    # context from graph capture, while start_load_kv() has already published a
    # newer active context on the connector.  Prefer the newest connector context
    # when it is active or has a higher generation.
    meta_context = getattr(attn_metadata, "sparse_kv_step_context", None)
    if meta_context is None:
        common_meta = getattr(attn_metadata, "common_metadata", None)
        if common_meta is not None:
            meta_context = getattr(common_meta, "sparse_kv_step_context", None)

    connector_context = None
    connector = _get_sparse_connector()
    getter = getattr(connector, "get_sparse_kv_step_context", None)
    if callable(getter):
        try:
            connector_context = getter(create_if_missing=False)
        except TypeError:
            connector_context = getter()
        except Exception:
            connector_context = None

    if connector_context is not None:
        if (
            meta_context is None
            or _ctx_host_active_reqs(connector_context) > 0
            or _ctx_generation(connector_context) > _ctx_generation(meta_context)
        ):
            return connector_context

    if meta_context is not None:
        return meta_context

    # CUDA graph capture can execute attention before a real SchedulerOutput has
    # installed request metadata.  Fetch/create the connector's persistent empty
    # context so the graph captures the sparse custom-op call with stable tensor
    # addresses.  This path must not clear a prepared active context; the getter
    # in vllm_v1_adapter.py preserves pending active contexts.
    if callable(getter):
        try:
            return getter(create_if_missing=True)
        except TypeError:
            return getter()
        except Exception:
            return None
    return None


class SparseSSDAttentionBackend(AttentionBackend):
    """Backend marker and vLLM metadata/KV-cache integration for AI-SSD."""

    # KV update is handled by vLLM's normal unified_kv_cache_update before
    # calling impl.forward().  Sparse selected-load then overwrites/populates
    # selected paged-cache blocks before sparse attention reads them.
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "SPARSE_SSD"

    @staticmethod
    def get_impl_cls() -> type["SparseSSDAttentionImpl"]:
        return SparseSSDAttentionImpl

    @staticmethod
    def get_builder_cls():
        # Reuse FlashAttention metadata builder.  The sparse backend consumes the
        # same CommonAttentionMetadata and attaches sparse metadata dynamically.
        return FlashAttentionBackend.get_builder_cls()

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=include_num_layers_dimension
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return FlashAttentionBackend.get_supported_head_sizes()

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return FlashAttentionBackend.supports_dtype(dtype)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return FlashAttentionBackend.supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return FlashAttentionBackend.supports_block_size(block_size)

    @classmethod
    def supports_sink(cls) -> bool:
        return FlashAttentionBackend.supports_sink()

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return FlashAttentionBackend.supports_alibi_sqrt()

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return FlashAttentionBackend.supports_mm_prefix()

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return FlashAttentionBackend.supports_per_head_quant_scales()

    @classmethod
    def supports_non_causal(cls) -> bool:
        return FlashAttentionBackend.supports_non_causal()

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return FlashAttentionBackend.supports_batch_invariance()

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return FlashAttentionBackend.supports_attn_type(attn_type)

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return FlashAttentionBackend.supports_compute_capability(capability)

    @classmethod
    def get_required_kv_cache_layout(cls):
        return FlashAttentionBackend.get_required_kv_cache_layout()

    @classmethod
    def is_sparse(cls) -> bool:
        # Keep this False because this is not vLLM's built-in sparse MLA path;
        # it is a full decoder attention backend replacement with sparse KV
        # loading inside its impl.forward().  Setting True would make normal
        # use_sparse=False selection reject the backend.
        return False


class SparseSSDAttentionImpl(AttentionImpl[AttentionMetadata]):
    """Q-aware sparse KV selection + sparse selected-block attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **extra_impl_args: Any,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        fallback_cls = FlashAttentionBackend.get_impl_cls()
        self._fallback_impl = fallback_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        self.supports_quant_query_input = getattr(
            self._fallback_impl, "supports_quant_query_input", False
        )
        self.can_return_lse_for_decode = getattr(
            self._fallback_impl, "can_return_lse_for_decode", False
        )
        self.need_to_return_lse_for_decode = getattr(
            self._fallback_impl, "need_to_return_lse_for_decode", False
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        return self._fallback_impl.process_weights_after_loading(act_dtype)

    def fused_output_quant_supported(self, quant_key):
        return self._fallback_impl.fused_output_quant_supported(quant_key)

    def fused_rope_kvcache_supported(self):
        return self._fallback_impl.fused_rope_kvcache_supported()

    def do_kv_cache_update(self, *args: Any, **kwargs: Any):
        return self._fallback_impl.do_kv_cache_update(*args, **kwargs)

    def _fallback_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._fallback_impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def _should_raise_on_missing_sparse(self) -> bool:
        """Fail closed for the production sparse-attention path.

        When SPARSE_SSD is selected because lmcache.enable_sparse_attention=true,
        falling back to FlashAttention silently turns the run back into the old
        full-KV path.  That is useful only for early debugging, not production.
        Therefore, once the LMCache sparse connector is available, either
        disable_full_load or enable_sparse_attention makes missing sparse
        context / missing custom op a hard error.
        """
        connector = _get_sparse_connector()
        spec = getattr(connector, "sparse_kv_spec", None) if connector is not None else None
        if spec is None:
            return False
        return bool(
            getattr(spec, "disable_full_load", False)
            or getattr(spec, "enable_sparse_attention", False)
        )

    def _write_sparse_fa_replay_marker(
        self,
        step_context: dict[str, Any],
        q_tokens: int,
    ) -> None:
        """Write a graph-captured marker proving FA-varlen replay metadata.

        Python forward is not re-entered during CUDA graph replay. When
        VLLM_SPARSE_FA_REPLAY_DEBUG=1, these tiny tensor writes are captured
        in the graph and replayed before FlashAttention runs. The LMCache
        connector drains fa_replay_debug_marker before preparing the next step.
        """
        if not _sparse_fa_replay_debug_enabled():
            return

        marker = step_context.get("fa_replay_debug_marker")
        fa_seq_lens = step_context.get("fa_seq_lens")
        fa_block_table = step_context.get("fa_block_table")
        fa_query_start_loc = step_context.get("fa_query_start_loc")
        active_reqs = step_context.get("active_reqs")
        selected_ready_flags = step_context.get("selected_ready_flags")
        selected_block_lens = step_context.get("selected_block_lens")

        if (
            not isinstance(marker, torch.Tensor)
            or marker.numel() < 8
            or not isinstance(fa_seq_lens, torch.Tensor)
            or fa_seq_lens.numel() < 1
            or not isinstance(fa_block_table, torch.Tensor)
            or fa_block_table.numel() < 1
            or not isinstance(fa_query_start_loc, torch.Tensor)
            or fa_query_start_loc.numel() < 2
        ):
            return

        # All operations below are CUDA-graph-capturable tensor ops. Do not use
        # .item(), .cpu(), or Python-side branching on tensor values here.
        marker.zero_()
        marker[0].copy_(fa_seq_lens[0].to(dtype=marker.dtype))
        marker[1].copy_(fa_block_table[0, 0].to(dtype=marker.dtype))
        if fa_block_table.dim() >= 2 and fa_block_table.shape[1] > 1:
            marker[2].copy_(fa_block_table[0, 1].to(dtype=marker.dtype))
        else:
            marker[2].fill_(-1)
        marker[3].copy_(fa_query_start_loc[1].to(dtype=marker.dtype))

        if isinstance(active_reqs, torch.Tensor):
            marker[4].copy_(active_reqs.reshape(()).to(dtype=marker.dtype))
        if isinstance(selected_ready_flags, torch.Tensor) and selected_ready_flags.numel() > 0:
            marker[5].copy_(selected_ready_flags[0].to(dtype=marker.dtype))
        marker[6].fill_(int(q_tokens))
        if isinstance(selected_block_lens, torch.Tensor) and selected_block_lens.numel() > 0:
            marker[7].copy_(selected_block_lens[0].to(dtype=marker.dtype))

    def _can_route_sparse_fa_varlen(
        self,
        query: torch.Tensor,
        step_context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Return whether this step is safe to run through sparse FA-varlen.

        FA-varlen sparse metadata is sized for decode-like sparse steps whose
        number of query rows equals the number of active requests.  Prefill and
        chunked-prefill can have thousands of query rows while no AISSD selector
        result is available yet; those steps must use the normal dense
        FlashAttention fallback.  This guard is intentionally placed before the
        AISSD selector op, so prefill/warmup does not issue HOST<->SSD RPC.
        """
        q_tokens = int(query.shape[0])
        try:
            host_active_reqs = int(step_context.get("host_active_reqs", 0) or 0)
        except Exception:
            host_active_reqs = 0
        if host_active_reqs <= 0:
            return False, f"no_active_sparse_requests(host_active_reqs={host_active_reqs})"

        fa_block_table = step_context.get("fa_block_table")
        fa_query_start_loc = step_context.get("fa_query_start_loc")
        if not isinstance(fa_block_table, torch.Tensor):
            return False, "missing_fa_block_table"
        if not isinstance(fa_query_start_loc, torch.Tensor):
            return False, "missing_fa_query_start_loc"

        block_rows = int(fa_block_table.shape[0]) if fa_block_table.dim() >= 1 else 0
        qstart_rows = int(fa_query_start_loc.shape[0]) if fa_query_start_loc.dim() >= 1 else 0
        if q_tokens + 1 > qstart_rows or q_tokens > block_rows:
            return (
                False,
                "metadata_capacity_prefill_or_large_batch("
                f"q_tokens={q_tokens}, block_table_rows={tuple(fa_block_table.shape)}, "
                f"query_start_loc={tuple(fa_query_start_loc.shape)}"
                ")",
            )
        return True, "ok"

    def _forward_sparse_fa_varlen(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        step_context: dict[str, Any],
        layer_name: str,
    ) -> torch.Tensor:
        """Run sparse selected-block attention through FlashAttention varlen.

        CUDA graph replay does not re-enter Python.  Therefore all tensors passed
        into FlashAttention here must be persistent graph-visible tensors whose
        contents are updated in-place by LMCacheConnector.prepare_sparse_kv_step().
        For the first implementation we support decode-like graph batches:
        max_query_len=1, causal=False, and selected blocks are interpreted as a
        compacted KV sequence ordered by original token position.
        """
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "VLLM_SPARSE_ATTENTION_IMPL=fa_varlen requested, but "
                "flash_attn_varlen_func is unavailable"
            )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "sparse FA-varlen path does not support fused output quantization"
            )

        prepare_t0 = time.perf_counter()
        q_tokens = int(query.shape[0])
        fa_block_table = step_context.get("fa_block_table")
        fa_seq_lens = step_context.get("fa_seq_lens")
        fa_query_start_loc = step_context.get("fa_query_start_loc")
        if (
            not isinstance(fa_block_table, torch.Tensor)
            or not isinstance(fa_seq_lens, torch.Tensor)
            or not isinstance(fa_query_start_loc, torch.Tensor)
        ):
            raise RuntimeError("sparse FA-varlen metadata tensors are missing")
        if q_tokens + 1 > int(fa_query_start_loc.shape[0]) or q_tokens > int(fa_block_table.shape[0]):
            raise RuntimeError(
                "sparse FA-varlen persistent metadata capacity is too small: "
                f"q_tokens={q_tokens}, block_table_rows={tuple(fa_block_table.shape)}, "
                f"query_start_loc={tuple(fa_query_start_loc.shape)}"
            )

        # Build a FlashAttention metadata object that binds the CUDA graph to
        # persistent sparse FA tensors.  If vLLM passes a real metadata object,
        # shallow-copy it and replace the dynamic fields.  If attn_metadata is
        # None during profiling / CUDA graph capture, still construct a minimal
        # FlashAttentionMetadata instead of returning output.fill_(0); otherwise
        # the captured graph would not contain a FlashAttention varlen op that
        # can consume runtime-updated selected-block metadata.
        sparse_block_table = fa_block_table[:q_tokens]
        sparse_seq_lens = fa_seq_lens[:q_tokens]
        sparse_query_start_loc = fa_query_start_loc[: q_tokens + 1]
        sparse_max_seq_len = int(step_context.get("fa_max_seq_len", 1) or 1)

        if attn_metadata is None:
            sparse_meta = FlashAttentionMetadata(
                num_actual_tokens=q_tokens,
                max_query_len=1,
                query_start_loc=sparse_query_start_loc,
                max_seq_len=sparse_max_seq_len,
                seq_lens=sparse_seq_lens,
                block_table=sparse_block_table,
                slot_mapping=torch.empty(0, dtype=torch.long, device=query.device),
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                max_dcp_context_kv_len=None,
                dcp_context_kv_lens=None,
                scheduler_metadata=None,
                prefix_scheduler_metadata=None,
                max_num_splits=0,
                causal=False,
            )
            if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
                logger.info(
                    "[sparse-attn-fa] built dummy metadata for capture/warmup "
                    "layer=%s q_shape=%s q_tokens=%d host_reqs=%s "
                    "selected_blocks=%s max_seq_len=%s generation=%s",
                    layer_name,
                    tuple(query.shape),
                    q_tokens,
                    step_context.get("host_active_reqs"),
                    step_context.get("host_selected_blocks"),
                    sparse_max_seq_len,
                    step_context.get("context_generation"),
                )
        else:
            sparse_meta = copy.copy(attn_metadata)
            sparse_meta.num_actual_tokens = q_tokens
            sparse_meta.max_query_len = 1
            sparse_meta.query_start_loc = sparse_query_start_loc
            sparse_meta.seq_lens = sparse_seq_lens
            sparse_meta.block_table = sparse_block_table
            sparse_meta.max_seq_len = sparse_max_seq_len
            sparse_meta.causal = False
            sparse_meta.use_cascade = False
            sparse_meta.common_prefix_len = 0
            sparse_meta.cu_prefix_query_lens = None
            sparse_meta.prefix_kv_lens = None
            sparse_meta.suffix_kv_lens = None
            sparse_meta.prefix_scheduler_metadata = None
            # Scheduler metadata produced for the original dense sequence does not
            # describe the compacted selected-block sequence, so disable it for the
            # sparse FA route.  FA will use its normal non-AOT scheduling.
            sparse_meta.scheduler_metadata = None
            sparse_meta.max_num_splits = 0

        if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
            logger.info(
                "[sparse-attn-fa] route layer=%s q_shape=%s q_tokens=%d "
                "host_reqs=%s selected_blocks=%s max_seq_len=%s generation=%s",
                layer_name,
                tuple(query.shape),
                q_tokens,
                step_context.get("host_active_reqs"),
                step_context.get("host_selected_blocks"),
                sparse_meta.max_seq_len,
                step_context.get("context_generation"),
            )

        self._write_sparse_fa_replay_marker(step_context, q_tokens)

        prepare_ms = (time.perf_counter() - prepare_t0) * 1000.0
        step_context["sparse_attn_prepare_ms"] = float(prepare_ms)
        if _aissd_selector_stats_enabled() and not torch.cuda.is_current_stream_capturing():
            logger.info(
                "[aissd-sparse-attn-prepare] layer=%s generation=%s selected_blocks=%s "
                "selected_tokens=%s q_tokens=%d fa_max_seq_len=%s prepare_ms=%.3f",
                layer_name,
                step_context.get("context_generation"),
                step_context.get("host_selected_blocks"),
                step_context.get("sparse_selected_tokens"),
                q_tokens,
                sparse_max_seq_len,
                prepare_ms,
            )

        fa_prof = _sparse_profile_begin("sparse_attention", layer_name, query)
        kernel_t0 = time.perf_counter()
        try:
            result = self._fallback_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                sparse_meta,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        finally:
            step_context["sparse_attention_kernel_ms"] = float((time.perf_counter() - kernel_t0) * 1000.0)
            _sparse_profile_end(
                fa_prof,
                step_context,
                query,
                impl="fa_varlen",
                extra=f"fa_max_seq_len={sparse_max_seq_len}",
            )
        return result

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_name = getattr(layer, "layer_name", "")
        step_context = _get_sparse_step_context(attn_metadata)

        if step_context is not None:
            # CUDA graph capture/replay must not perform GPU->CPU syncs or
            # allocate temporary CUDA tensors.  In particular, do not call
            # .item() on active_reqs here and do not use torch.zeros(...) as a
            # dict.get default value, because Python evaluates defaults eagerly.
            if _sparse_kv_debug_enabled() and not torch.cuda.is_current_stream_capturing():
                active_reqs_obj = step_context.get("active_reqs", None)
                if isinstance(active_reqs_obj, torch.Tensor):
                    active_reqs_for_log = int(active_reqs_obj.detach().cpu().item())
                else:
                    active_reqs_for_log = len(step_context.get("req_ids", []))
                logger.info(
                    "[sparse-attn] custom-op route layer=%s q_shape=%s reqs=%d "
                    "host_reqs=%s selected_blocks=%s generation=%s ctx_id=%s",
                    layer_name,
                    tuple(query.shape),
                    active_reqs_for_log,
                    step_context.get("host_active_reqs"),
                    step_context.get("host_selected_blocks"),
                    step_context.get("context_generation"),
                    hex(id(step_context)),
                )
            _log_sparse_context_ptr("attn-before-op", layer_name, step_context)
            try:
                impl = _sparse_attention_impl()
                if impl in ("fa_varlen", "flash", "flash_attn", "flashattention"):
                    can_route_fa, skip_reason = self._can_route_sparse_fa_varlen(
                        query, step_context
                    )
                    if not can_route_fa:
                        if (
                            _sparse_attention_diag_enabled()
                            or _sparse_kv_debug_enabled()
                        ) and not torch.cuda.is_current_stream_capturing():
                            logger.info(
                                "[sparse-attn-fa] skip sparse FA-varlen route "
                                "layer=%s q_shape=%s generation=%s reason=%s; "
                                "falling back to FlashAttention",
                                layer_name,
                                tuple(query.shape),
                                step_context.get("context_generation"),
                                skip_reason,
                            )
                        return self._fallback_forward(
                            layer,
                            query,
                            key,
                            value,
                            kv_cache,
                            attn_metadata,
                            output,
                            output_scale,
                            output_block_scale,
                        )

                sparse_total_prof = _sparse_profile_begin("sparse_forward_total", layer_name, query)
                selector_ms = _maybe_run_aissd_selector_op(
                    query=query,
                    step_context=step_context,
                    layer_name=layer_name,
                    head_size=int(self.head_size),
                    num_heads=int(self.num_heads),
                    num_kv_heads=int(self.num_kv_heads),
                )
                if impl in ("fa_varlen", "flash", "flash_attn", "flashattention"):
                    attn_t0 = time.perf_counter()
                    result = self._forward_sparse_fa_varlen(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                        step_context,
                        layer_name,
                    )
                    attention_ms = (time.perf_counter() - attn_t0) * 1000.0
                    step_context["sparse_attention_ms"] = float(attention_ms)
                    _log_sparse_kv_e2e_bandwidth(
                        step_context=step_context,
                        layer_name=layer_name,
                        impl="fa_varlen",
                        q_shape=tuple(query.shape),
                        selector_ms=float(selector_ms),
                        attention_ms=float(attention_ms),
                    )
                    _sparse_profile_end(
                        sparse_total_prof, step_context, query, impl="fa_varlen"
                    )
                    return result
                if impl not in ("custom", "cuda", "sparse_flash"):
                    raise RuntimeError(
                        f"Unknown VLLM_SPARSE_ATTENTION_IMPL={impl!r}; "
                        "expected custom or fa_varlen"
                    )
                # Keep debug_counters graph-visible even during CUDA graph
                # capture.  Python does not re-enter this forward() on graph
                # replay, so disabling counters during capture would make the
                # replayed sparse_flash_attention kernel permanently run with a
                # zero-length counter tensor.  The captured graph safely zeroes
                # and updates this stable device tensor on every replay;
                # LMCacheConnector.prepare_sparse_kv_step() drains the previous
                # replay's values before publishing the next step context.
                debug_enabled = _sparse_debug_counters_enabled()
                debug_counters = _ensure_sparse_debug_counters(
                    step_context, query.device, debug_enabled
                )
                if debug_enabled:
                    debug_counters.zero_()

                custom_prof = _sparse_profile_begin("sparse_attention", layer_name, query)
                attn_t0 = time.perf_counter()
                vllm_ops.sparse_flash_attention(
                    output,
                    query,
                    kv_cache,
                    step_context.get("active_reqs"),
                    step_context["req_token_lens"],
                    step_context["req_vllm_cached_tokens"],
                    step_context["req_lmcache_cached_tokens"],
                    step_context["req_slot_lens"],
                    step_context["slot_mapping_table"],
                    step_context["selected_block_table"],
                    step_context["selected_block_lens"],
                    step_context["selected_ready_flags"],
                    debug_counters,
                    int(step_context["block_size"]),
                    int(step_context["chunk_size"]),
                    int(step_context["top_n_chunks"]),
                    float(self.scale),
                )
                attention_ms = (time.perf_counter() - attn_t0) * 1000.0
                step_context["sparse_attention_ms"] = float(attention_ms)
                _sparse_profile_end(
                    custom_prof, step_context, query, impl="custom"
                )
                _log_sparse_kv_e2e_bandwidth(
                    step_context=step_context,
                    layer_name=layer_name,
                    impl="custom",
                    q_shape=tuple(query.shape),
                    selector_ms=float(selector_ms),
                    attention_ms=float(attention_ms),
                )
                if debug_enabled and not torch.cuda.is_current_stream_capturing():
                    (
                        launched_qh_blocks,
                        active_qh_blocks,
                        selected_block_visits,
                        selected_token_visits,
                        inactive_qh_blocks,
                        invalid_block_refs,
                        kernel_active_reqs,
                        kernel_max_selected_len,
                    ) = _read_sparse_debug_counters(debug_counters)
                    logger.info(
                        "[sparse-attn-kernel] layer=%s q_shape=%s "
                        "launched_qh_blocks=%d active_qh_blocks=%d "
                        "inactive_qh_blocks=%d selected_block_visits=%d "
                        "selected_token_visits=%d invalid_block_refs=%d "
                        "kernel_active_reqs=%d kernel_max_selected_len=%d "
                        "host_reqs=%s host_selected_blocks=%s generation=%s",
                        layer_name,
                        tuple(query.shape),
                        launched_qh_blocks,
                        active_qh_blocks,
                        inactive_qh_blocks,
                        selected_block_visits,
                        selected_token_visits,
                        invalid_block_refs,
                        kernel_active_reqs,
                        kernel_max_selected_len,
                        step_context.get("host_active_reqs"),
                        step_context.get("host_selected_blocks"),
                        step_context.get("context_generation"),
                    )
                _sparse_profile_end(
                    sparse_total_prof, step_context, query, impl="custom"
                )
                return output
            except Exception:
                logger.exception(
                    "[sparse-attn] production sparse attention path failed "
                    "for layer=%s",
                    layer_name,
                )
                if self._should_raise_on_missing_sparse():
                    raise

        connector = _get_sparse_connector()
        reason_parts = []
        if step_context is None:
            reason_parts.append("no_sparse_step_context")
        if connector is None:
            reason_parts.append("no_registered_lmcache_connector")
        else:
            spec = getattr(connector, "sparse_kv_spec", None)
            reason_parts.append(f"connector={type(connector).__name__}")
            reason_parts.append(f"spec_enabled={getattr(spec, 'enabled', None)}")
            reason_parts.append(f"disable_full_load={getattr(spec, 'disable_full_load', None)}")
            getter = getattr(connector, "get_sparse_kv_step_context", None)
            reason_parts.append(f"has_context_getter={callable(getter)}")
        reason = ",".join(reason_parts) if reason_parts else "unknown"
        message = (
            f"[sparse-attn] production sparse context/custom op unavailable "
            f"for layer={layer_name}; reason={reason}; "
            f"impl={_sparse_attention_impl()}"
        )
        if self._should_raise_on_missing_sparse():
            raise RuntimeError(
                message
                + "; refusing FlashAttention fallback because "
                + "lmcache.enable_sparse_attention is enabled"
            )
        if _sparse_attention_diag_enabled():
            logger.warning(message + "; falling back to FlashAttention")
        else:
            logger.warning_once(message + "; falling back to FlashAttention")
        return self._fallback_forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
