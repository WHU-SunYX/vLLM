# SPDX-License-Identifier: Apache-2.0
"""AI-SSD sparse attention backend for vLLM V1.

This backend is wired into vLLM's normal attention backend selection path.
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
import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
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
                return output
            except Exception:
                logger.exception(
                    "[sparse-attn] production sparse_flash_attention custom op "
                    "failed for layer=%s",
                    layer_name,
                )
                if self._should_raise_on_missing_sparse():
                    raise

        message = (
            f"[sparse-attn] production sparse context/custom op unavailable "
            f"for layer={layer_name}"
        )
        if self._should_raise_on_missing_sparse():
            raise RuntimeError(
                message
                + "; refusing FlashAttention fallback because "
                + "lmcache.enable_sparse_attention is enabled"
            )
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
