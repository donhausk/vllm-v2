# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    cu_tokens_across_dp_cpu: torch.Tensor


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    # Holds attention statistics for the current forward pass as tensors
    attn_stats: Optional[Dict[str, torch.Tensor]] = None

    vllm_config: Optional[VllmConfig] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1:
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if attn_metadata is not None:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
        else:
            batchsize = num_tokens
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = batchsize
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)
        dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        vllm_config=vllm_config) # Required for the attention type

    # Populate attention stats if metadata is available
    if _forward_context.attn_metadata is not None:
        _forward_context.attn_stats = {}
        # Try fetching token counts, handle different backend versions/attributes
        # Ensure stats are stored as tensors on the correct device
        device = _forward_context.attn_metadata.device if hasattr(_forward_context.attn_metadata, 'device') else 'cuda' # Default device
        try:
            # Common attributes in AttentionMetadata
            num_prefill = getattr(_forward_context.attn_metadata, "num_prefills", 0)
            num_decode = getattr(_forward_context.attn_metadata, "num_decodes", 0)

            num_prefill_tokens = 0
            num_decode_tokens = 0

            if num_prefill > 0:
                 num_prefill_tokens = getattr(_forward_context.attn_metadata, "num_prefill_tokens", 0)
            if num_decode > 0:
                 num_decode_tokens = getattr(_forward_context.attn_metadata, "num_decode_tokens", 0)

            # Fallback or alternative for v1 backends if specific attrs aren't present
            if num_prefill == 0 and num_decode == 0:
                 is_prompt = getattr(_forward_context.attn_metadata, "is_prompt", False) # Need heuristic if is_prompt is not direct attr
                 num_tokens_fallback = getattr(_forward_context.attn_metadata, "num_input_tokens", 0)
                 if is_prompt: # Simplified assumption: all tokens are prefill if it's a prompt phase
                     num_prefill_tokens = num_tokens_fallback
                 else:
                     num_decode_tokens = num_tokens_fallback

            _forward_context.attn_stats["num_prefill_tokens"] = torch.tensor(num_prefill_tokens, device=device, dtype=torch.long)
            _forward_context.attn_stats["num_decode_tokens"] = torch.tensor(num_decode_tokens, device=device, dtype=torch.long)

        except AttributeError as e:
            # Log if expected attributes are missing, but don't crash
            logger.debug(f"Could not populate token counts in AttentionStats: {e}")
            # Initialize with zeros if attributes are missing
            _forward_context.attn_stats["num_prefill_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
            _forward_context.attn_stats["num_decode_tokens"] = torch.tensor(0, device=device, dtype=torch.long)

    # KVConnector: trigger (possibly async) load before forward.
    # Each attn layer will block until the reading is complete.
    trigger_kv_transfer = (attn_metadata is not None
                           and has_kv_transfer_group()
                           and is_v1_kv_transfer_group())
    if trigger_kv_transfer:
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        kv_connector.start_load_kv(_forward_context)

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        # KVConnector: each attn layer triggers (possibly async) save.
        # Ensure all those operations complete before forward() is done.
        if trigger_kv_transfer:
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            kv_connector.wait_for_save()

        _forward_context = prev_context
