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


def staged_prefill_enabled(config):
    transfer = getattr(config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None) or {}
    mode = extra.get("lmcache.aissd_gpu_prefill_mode", "legacy")
    if mode not in ("legacy", "staged"):
        raise ValueError("AISSD prefill mode must be legacy or staged")
    return mode == "staged"


def staged_prefill_prefetch_enabled(config):
    transfer = getattr(config, "kv_transfer_config", None)
    extra = getattr(transfer, "kv_connector_extra_config", None) or {}
    enabled = extra.get("lmcache.aissd_gpu_prefill_prefetch", False)
    if type(enabled) is not bool:
        raise ValueError("lmcache.aissd_gpu_prefill_prefetch must be a JSON boolean")
    return enabled


def staged_prefill_live_blocks(config):
    """Max live pages of one prefill step, excluding the shared null page."""
    _, chunk, _ = resident_options(config)
    block_size = config.cache_config.block_size
    tokens = min(config.model_config.max_model_len,
                 config.scheduler_config.max_num_batched_tokens + chunk - 1)
    return (tokens + block_size - 1) // block_size


def staged_prefill_pool_blocks(config):
    budget, _, _ = resident_options(config)
    return 1 + max(staged_prefill_live_blocks(config), budget // config.cache_config.block_size)


def validate_resident_config(config):
    budget, chunk, topn = resident_options(config)
    staged = staged_prefill_enabled(config)
    if staged_prefill_prefetch_enabled(config) and not staged:
        raise ValueError("AISSD prefill prefetch requires aissd_gpu_prefill_mode=staged")
    if budget == 0:
        if staged:
            raise ValueError("AISSD staged prefill requires a positive resident budget")
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
    if staged:
        if not scheduler.enable_chunked_prefill:
            invalid.append("--enable-chunked-prefill")
        if config.model_config.max_model_len <= 0 or getattr(
                config.model_config, "original_max_model_len", None) == -1:
            invalid.append("a fixed positive --max-model-len")
        if config.kv_transfer_config.kv_role != "kv_both":
            invalid.append("kv_role=kv_both")
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
    bootstrap: bool = True


@dataclass
class PrefillBlocks:
    prefix_tokens: int
    external_tail_tokens: int
    computed_before: int
    total_tokens: int
    released_blocks: int


class ResidentBlockManagerMixin:
    def configure_aissd_residency(self, config):
        self._aissd_budget, self._aissd_chunk, self._aissd_topn = resident_options(config)
        self._aissd_ready = {}
        self._aissd_states = {}
        self._aissd_epoch = 0
        validate_resident_config(config)
        self._aissd_staged_prefill = staged_prefill_enabled(config)
        self._aissd_prefill_states = {}
        if self._aissd_budget:
            managers = self.coordinator.single_type_managers
            if len(managers) != 1 or type(managers[0].kv_cache_spec).__name__ != "FullAttentionSpec":
                raise ValueError("AISSD GPU residency supports one FullAttentionSpec group")
            if str(managers[0].kv_cache_spec.dtype) != "torch.bfloat16":
                raise ValueError("AISSD GPU residency currently requires BF16 KV")
        if self._aissd_staged_prefill:
            self._aissd_prefill_live_blocks = staged_prefill_live_blocks(config)
            if self.block_pool.get_num_free_blocks() < staged_prefill_pool_blocks(config) - 1:
                raise ValueError("AISSD KV pool cannot cover the staged prefill/decode page budget")

    def aissd_mark_archive_ready(self, ready):
        for req_id, prefix in ready.items():
            prefix = int(prefix)
            old = self._aissd_ready.get(req_id, prefix)
            staged = getattr(self, "_aissd_staged_prefill", False)
            if staged and (prefix <= 0 or prefix % self._aissd_chunk or prefix < old):
                raise RuntimeError("Invalid staged prefill archive acknowledgement")
            if old != prefix and (not staged or req_id in self._aissd_states):
                raise RuntimeError("Immutable resident SSD prefix changed")
            self._aissd_ready[req_id] = prefix

    @staticmethod
    def _aissd_check_owned_pages(pages):
        if (any(b.is_null or b.ref_cnt != 1 or b.block_hash is not None for b in pages)
                or len({b.block_id for b in pages}) != len(pages)):
            raise RuntimeError("AISSD cannot release shared, null or invalid KV pages")

    def allocate_aissd_prefill_slots(self, request, num_new_tokens, external_tokens):
        """Keep the logical address space while allocating only the live suffix."""
        manager = self.coordinator.single_type_managers[0]
        req_id = request.request_id
        block_size, chunk = manager.block_size, self._aissd_chunk
        state = self._aissd_prefill_states.get(req_id)
        blocks = manager.req_to_blocks[req_id]
        computed = request.num_computed_tokens + external_tokens
        total = computed + num_new_tokens
        if not 0 <= computed < total <= min(self.max_model_len, request.num_tokens):
            raise RuntimeError("Invalid staged prefill schedule")
        if state is None:
            if blocks or request.num_computed_tokens:
                raise RuntimeError("AISSD staged prefill started with untracked local KV")
            if request.num_tokens != request.num_prompt_tokens:
                raise RuntimeError("AISSD staged prefill cannot resume a preempted decode")
            prefix = computed // chunk * chunk
            partial = computed - prefix
            if partial and (partial != chunk - 1 or computed + 1 != request.num_prompt_tokens):
                raise RuntimeError("Only the full-hit recomputed tail may split an external KV chunk")
            released = 0
            old_prefix = prefix
            retiring = []
        else:
            if external_tokens or computed != state.total_tokens:
                raise RuntimeError("AISSD staged prefill was rewound without clearing its pages")
            old_prefix = state.prefix_tokens
            prefix = min(self._aissd_ready.get(req_id, 0), computed // chunk * chunk)
            if prefix < old_prefix or prefix > len(blocks) * block_size:
                raise RuntimeError("AISSD staged prefill archive does not cover the released pages")
            retiring = blocks[old_prefix // block_size:prefix // block_size]
            self._aissd_check_owned_pages(retiring)
            partial = 0
            released = state.released_blocks + len(retiring)

        end = (total + block_size - 1) // block_size
        prefix_pages = prefix // block_size
        if end - prefix_pages > self._aissd_prefill_live_blocks:
            raise RuntimeError("Staged prefill live page bound exceeded")
        new_count = end - (len(blocks) if state is not None else prefix_pages)
        if new_count < 0 or new_count > self.block_pool.get_num_free_blocks() + len(retiring):
            raise RuntimeError("AISSD staged prefill cannot allocate its active window")
        if state is None:
            blocks.extend([self.block_pool.null_block] * prefix_pages)
        elif retiring:
            blocks[old_prefix // block_size:prefix_pages] = [self.block_pool.null_block] * len(retiring)
            self.block_pool.free_blocks(reversed(retiring))
        new_blocks = manager.allocate_new_blocks(req_id, total, total)
        self._aissd_prefill_states[req_id] = PrefillBlocks(prefix, partial, computed, total, released)
        return self.create_kv_cache_blocks((new_blocks,))

    def prepare_aissd_residency(self, request, num_new_tokens):
        budget = getattr(self, "_aissd_budget", 0)
        if not budget:
            return
        req_id = request.request_id
        prefix = self._aissd_ready.get(req_id)
        if request.num_computed_tokens < request.num_prompt_tokens:
            return
        prefill = getattr(self, "_aissd_prefill_states", {}).get(req_id)
        if prefill is not None:
            expected = request.num_prompt_tokens // self._aissd_chunk * self._aissd_chunk
            if (prefix or 0) != expected:
                raise RuntimeError("AISSD staged prefill has not committed the final prompt prefix")
        if not prefix:
            if (getattr(self, "_aissd_staged_prefill", False)
                    and request.num_computed_tokens + num_new_tokens > budget):
                raise RuntimeError("AISSD decode exceeds the resident budget without an archived prefix")
            getattr(self, "_aissd_prefill_states", {}).pop(req_id, None)
            return
        manager = self.coordinator.single_type_managers[0]
        blocks = manager.req_to_blocks[req_id]
        total = request.num_computed_tokens + num_new_tokens
        keep = hot_block_count(budget, prefix, total, manager.block_size,
                               self._aissd_chunk, self._aissd_topn)
        state = self._aissd_states.get(req_id)
        if state is None:
            prefix_count = prefix // manager.block_size
            if prefill is not None:
                prior = prefill.prefix_tokens // manager.block_size
                if not prior <= prefix_count <= len(blocks) or any(not b.is_null for b in blocks[:prior]):
                    raise RuntimeError("AISSD staged decode prefix differs from its committed archive")
                retiring = blocks[prior:prefix_count]
                self._aissd_check_owned_pages(retiring)
                new_count = max(0, (total + manager.block_size - 1) // manager.block_size - len(blocks))
                if keep + new_count > self.block_pool.get_num_free_blocks() + len(retiring):
                    raise RuntimeError("AISSD cannot reserve the decode hot pages")
                blocks[prior:prefix_count] = [self.block_pool.null_block] * len(retiring)
                self.block_pool.free_blocks(reversed(retiring))
                hot = self.block_pool.get_new_blocks(keep)
                self._aissd_epoch += 1
                self._aissd_states[req_id] = ResidentBlocks(
                    self._aissd_epoch, prefix, hot, prefill.released_blocks + len(retiring), False)
                self._aissd_prefill_states.pop(req_id)
                return
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
                prefill = getattr(self, "_aissd_prefill_states", {}).get(req_id)
                if prefill is not None:
                    table = list(self.get_block_ids(req_id)[0])
                    block_size = self.coordinator.single_type_managers[0].block_size
                    prefix_pages = prefill.prefix_tokens // block_size
                    result[req_id] = {
                        "prefill": True, "prefix_tokens": prefill.prefix_tokens,
                        "external_tail_tokens": prefill.external_tail_tokens,
                        "computed_before": prefill.computed_before, "total_tokens": prefill.total_tokens,
                        "logical_block_ids": table, "released_blocks": prefill.released_blocks,
                        "owned_blocks": len(table) - prefix_pages,
                    }
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
                "bootstrap": state.bootstrap,
            }
        return result

    def free_aissd_residency(self, req_id):
        if not getattr(self, "_aissd_budget", 0):
            return
        state = self._aissd_states.pop(req_id, None)
        if state is not None:
            self.block_pool.free_blocks(reversed(state.hot_blocks))
        self._aissd_ready.pop(req_id, None)
        getattr(self, "_aissd_prefill_states", {}).pop(req_id, None)
        if getattr(self, "_aissd_staged_prefill", False):
            manager = self.coordinator.single_type_managers[0]
            pages = manager.req_to_blocks.get(req_id)
            if pages:
                # The coordinator subsequently frees the remaining real pages.
                # Null entries only encode logical positions; they own no page.
                pages[:] = [page for page in pages if not page.is_null]
