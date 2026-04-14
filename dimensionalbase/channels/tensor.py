"""
TensorChannel — Channel 3. Lossless. Future.

Available when: agents share GPU hardware.
What transfers: KV cache slices, raw model states.
Information loss: zero.
Speed: microseconds (RDMA).

This is a stub for Phase 4. The architecture is in place so that
when tensor sharing lands, it slides in without changing agent code.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional

from dimensionalbase.channels.base import Channel
from dimensionalbase.core.entry import KnowledgeEntry
from dimensionalbase.core.types import ChannelCapability, ChannelLevel

logger = logging.getLogger("dimensionalbase.channels.tensor")


class TensorChannel(Channel):
    """Channel 3: Tensor/KV-cache sharing (future implementation).

    This channel will support:
    - KV cache slice sharing between agents on the same GPU
    - CUDA IPC for inter-process tensor transfer
    - TransferEngine integration for multi-node GPU clusters
    - Sub-symbolic communication (share reasoning state, not text)

    For now, this is a placeholder that reports as unavailable.
    """

    def __init__(self):
        self._available = self._detect_gpu()
        if self._available:
            logger.info("TensorChannel: GPU detected, but implementation pending (Phase 4)")
        else:
            logger.debug("TensorChannel: no GPU detected")

    def store(self, entry: KnowledgeEntry) -> None:
        raise NotImplementedError("TensorChannel is planned for Phase 4")

    def retrieve(self, path: str) -> Optional[KnowledgeEntry]:
        raise NotImplementedError("TensorChannel is planned for Phase 4")

    def query_by_path(self, pattern: str) -> List[KnowledgeEntry]:
        raise NotImplementedError("TensorChannel is planned for Phase 4")

    def delete(self, path: str) -> bool:
        raise NotImplementedError("TensorChannel is planned for Phase 4")

    def all_entries(self) -> List[KnowledgeEntry]:
        raise NotImplementedError("TensorChannel is planned for Phase 4")

    def count(self) -> int:
        return 0

    def capability(self) -> ChannelCapability:
        return ChannelCapability(
            level=ChannelLevel.TENSOR,
            available=False,
            description="Tensor/KV-cache sharing. Planned for Phase 4.",
            latency_estimate="microseconds (RDMA)",
            information_loss="zero",
        )

    def close(self) -> None:
        pass

    @staticmethod
    def _detect_gpu() -> bool:
        """Check if CUDA GPUs are available.

        The tensor channel is still a stub, so startup should not pay the cost
        of importing heavy ML stacks or probing external binaries unless the
        caller explicitly opts in.
        """
        if os.environ.get("DMB_ENABLE_TENSOR_PROBE") != "1":
            return False

        if shutil.which("nvidia-smi") is not None:
            return True

        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        except Exception:
            logger.debug("TensorChannel GPU probe failed", exc_info=True)
        return False
