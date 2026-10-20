# SPDX-License-Identifier: Apache-2.0
"""Eager, single-request GPU residency for an immutable SSD KV prefix.

The logical table retains null placeholders. Hot blocks have a separate owner
and must never be treated as prefix-cache hits. No GPU tensor is allocated here.
"""

from dataclasses import dataclass


def resident_options(config):
    transfer = getattr(config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None) or {}
    budget = int(extra.get("lmcache.aissd_gpu_resident_tokens", 0))
    chunk = int(extra.get("lmcache.aissd_gpu_resident_chunk_size", 128))
    topn = int(extra.get("lmcache.sparse_kv_top_n_chunks", 0))
    return budget, chunk, topn


def validate_resident_config(config):
    budget, chunk, topn = resident_options(config)
    if budget == 0:
        return
    if budget < 0 or chunk <= 0 or budget % chunk or topn <= 0:
        raise ValueError("AISSD resident budget must be positive, chunk-aligned; Top-N > 0")
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    extra = config.kv_transfer_config.kv_connector_extra_config or {}
    checks = {
        "--enforce-eager": config.model_config.enforce_eager,
        "--max-num-seqs 1": scheduler.max_num_seqs == 1,
        "--no-async-scheduling": not scheduler.async_scheduling,
        "VLLM_USE_V2_MODEL_RUNNER=0": not config.use_v2_model_runner,
        "--no-enable-prefix-caching": not cache.enable_prefix_caching,
        "single GPU (TP/PP/DCP/PCP/DP=1)": all(
            getattr(parallel, name, 1) == 1
            for name in ("tensor_parallel_size", "pipeline_parallel_size",
                         "decode_context_parallel_size", "prefill_context_parallel_size",
                         "data_parallel_size")
        ),
        "no speculative decoding": getattr(config, "speculative_config", None) is None,
        "text-only model": not getattr(config.model_config, "is_multimodal_model", False)
        and not getattr(config.model_config, "is_encoder_decoder", False),
        "external LMCache adapter": not extra.get("use_native", False),
        "LMCacheConnectorV1/Dynamic": config.kv_transfer_config.kv_connector
        in ("LMCacheConnectorV1", "LMCacheConnectorV1Dynamic"),
        "chunk multiple of block size": cache.block_size > 0 and chunk % cache.block_size == 0,
        "one complete selected set plus one local block": budget >= topn * chunk + cache.block_size,
    }
    invalid = [label for label, ok in checks.items() if not ok]
    if invalid:
        raise ValueError("AISSD GPU residency requires: " + "; ".join(invalid))


def hot_block_count(budget, prefix, total_tokens, block_size, chunk_size, topn):
    if prefix <= 0 or prefix % chunk_size or chunk_size % block_size:
        raise ValueError("Resident SSD prefix must contain complete, block-aligned chunks")
    if total_tokens < prefix:
        raise RuntimeError("Resident request rolled back into its offloaded prefix")
    local = (total_tokens - prefix + block_size - 1) // block_size
    per_chunk = chunk_size // block_size
    available = max(0, budget // block_size - local) // per_chunk
    required = min(topn, prefix // chunk_size)
    if available < required:
        raise RuntimeError(
            f"AISSD resident budget exhausted: local_blocks={local}, "
            f"required_selected_chunks={required}, budget_tokens={budget}. "
            "Increase the budget or reduce the generation length."
        )
    return min(available, prefix // chunk_size) * per_chunk


@dataclass
class ResidentBlocks:
    epoch: int
    prefix_tokens: int
    hot_blocks: list
    released_blocks: int = 0


class ResidentBlockManagerMixin:
    def configure_aissd_residency(self, config):
        self._aissd_budget, self._aissd_chunk, self._aissd_topn = resident_options(config)
        self._aissd_ready = {}
        self._aissd_states = {}
        self._aissd_epoch = 0
        validate_resident_config(config)
        if self._aissd_budget:
            managers = self.coordinator.single_type_managers
            if len(managers) != 1 or type(managers[0].kv_cache_spec).__name__ != "FullAttentionSpec":
                raise ValueError("AISSD GPU residency supports one FullAttentionSpec group")
            if str(managers[0].kv_cache_spec.dtype) != "torch.bfloat16":
                raise ValueError("AISSD GPU residency currently requires BF16 KV")

    def aissd_mark_archive_ready(self, ready):
        for req_id, prefix in ready.items():
            prefix = int(prefix)
            old = self._aissd_ready.get(req_id, prefix)
            if old != prefix:
                raise RuntimeError("Immutable resident SSD prefix changed")
            self._aissd_ready[req_id] = prefix

    def prepare_aissd_residency(self, request, num_new_tokens):
        budget = getattr(self, "_aissd_budget", 0)
        if not budget:
            return
        req_id = request.request_id
        prefix = self._aissd_ready.get(req_id)
        if not prefix or request.num_computed_tokens < request.num_prompt_tokens:
            return
        manager = self.coordinator.single_type_managers[0]
        blocks = manager.req_to_blocks[req_id]
        total = request.num_computed_tokens + num_new_tokens
        keep = hot_block_count(budget, prefix, total, manager.block_size,
                               self._aissd_chunk, self._aissd_topn)
        state = self._aissd_states.get(req_id)
        if state is None:
            prefix_count = prefix // manager.block_size
            original = blocks[:prefix_count]
            if len(original) != prefix_count or any(
                b.is_null or b.ref_cnt != 1 or b.block_hash is not None for b in original
            ) or len({b.block_id for b in original}) != prefix_count:
                raise RuntimeError("Cannot release shared, incomplete, or invalid resident KV prefix")
            self._aissd_epoch += 1
            state = ResidentBlocks(self._aissd_epoch, prefix, list(original[:keep]),
                                   prefix_count - keep)
            # Ownership of kept blocks moves to state; all logical prefix pages
            # become null. The original reference is freed exactly once.
            blocks[:prefix_count] = [self.block_pool.null_block] * prefix_count
            self.block_pool.free_blocks(reversed(original[keep:]))
            self._aissd_states[req_id] = state
        elif keep < len(state.hot_blocks):
            removed = state.hot_blocks[keep:]
            self.block_pool.free_blocks(reversed(removed))
            del state.hot_blocks[keep:]
            state.released_blocks += len(removed)

    def aissd_resident_snapshots(self, req_ids):
        if not getattr(self, "_aissd_budget", 0):
            return {}
        result = {}
        for req_id in req_ids:
            state = self._aissd_states.get(req_id)
            if state is None:
                continue
            table = list(self.get_block_ids(req_id)[0])
            prefix_blocks = state.prefix_tokens // self.coordinator.single_type_managers[0].block_size
            result[req_id] = {
                "epoch": state.epoch, "prefix_tokens": state.prefix_tokens,
                "hot_block_ids": [b.block_id for b in state.hot_blocks],
                "logical_block_ids": table, "released_blocks": state.released_blocks,
                "owned_blocks": len(state.hot_blocks) + len(table) - prefix_blocks,
                "pool_free_blocks": self.block_pool.get_num_free_blocks(),
                "budget_tokens": self._aissd_budget,
            }
        return result

    def free_aissd_residency(self, req_id):
        if not getattr(self, "_aissd_budget", 0):
            return
        state = self._aissd_states.pop(req_id, None)
        if state is not None:
            self.block_pool.free_blocks(reversed(state.hot_blocks))
        self._aissd_ready.pop(req_id, None)
