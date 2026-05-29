# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _sparse_ssd_attention_requested(vllm_config) -> bool:
    """Return true when LMCache sparse SSD attention should own the backend.

    This runs during Attention construction, before LMCache worker services are
    fully initialized, so the reliable source is vLLM's kv_connector_extra_config
    plus an explicit environment override.
    """
    env_value = getattr(envs, "LMCACHE_ENABLE_SPARSE_ATTENTION", None)
    # vllm.envs may not expose the custom env var; read os.environ lazily.
    if env_value is None:
        import os

        env_value = os.environ.get("LMCACHE_ENABLE_SPARSE_ATTENTION")
    if env_value is not None:
        return _as_bool(env_value, False)

    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is None:
        return False
    kv_extra = getattr(kv_cfg, "kv_connector_extra_config", None) or {}
    for key in (
        "lmcache.enable_sparse_attention",
        "enable_sparse_attention",
        "lmcache.sparse_attention",
        "sparse_attention",
    ):
        if key in kv_extra:
            return _as_bool(kv_extra.get(key), False)
    return False



class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False
    use_batch_invariant: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant})"
        )


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
    )

    if _sparse_ssd_attention_requested(vllm_config):
        from vllm.v1.attention.backends.sparse_ssd_attn import (
            SparseSSDAttentionBackend,
        )

        invalid_reasons = SparseSSDAttentionBackend.validate_configuration(
            head_size=attn_selector_config.head_size,
            dtype=attn_selector_config.dtype,
            kv_cache_dtype=attn_selector_config.kv_cache_dtype,
            block_size=attn_selector_config.block_size,
            use_mla=attn_selector_config.use_mla,
            has_sink=attn_selector_config.has_sink,
            use_sparse=False,
            use_mm_prefix=attn_selector_config.use_mm_prefix,
            use_per_head_quant_scales=attn_selector_config.use_per_head_quant_scales,
            device_capability=current_platform.get_device_capability(),
            attn_type=attn_selector_config.attn_type,
            use_non_causal=attn_selector_config.use_non_causal,
            use_batch_invariant=attn_selector_config.use_batch_invariant,
        )
        if invalid_reasons:
            logger.warning(
                "SPARSE_SSD attention requested but unsupported for %s: %s; "
                "falling back to platform backend",
                attn_selector_config,
                ", ".join(invalid_reasons),
            )
        else:
            logger.info("Using SPARSE_SSD attention backend due to LMCache sparse attention config")
            return SparseSSDAttentionBackend

    return _cached_get_attn_backend(
        backend=vllm_config.attention_config.backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, MambaAttentionBackendEnum)

    mamba_attn_backend = mamba_type.get_class()
    if envs.VLLM_BATCH_INVARIANT and not mamba_attn_backend.supports_batch_invariance():
        raise RuntimeError(
            "VLLM batch_invariant mode is not supported for "
            f"{mamba_attn_backend.get_name()}."
        )
    return mamba_attn_backend
