"""
Provenance Tracking — full lineage DAG for every piece of knowledge.

Every fact in DimensionalBase has a history. You can trace any knowledge
entry back to its origins through a directed acyclic graph:

  - Who wrote it, and when
  - What it was derived from
  - How it was transformed
  - Who confirmed or contradicted it
  - Every version that ever existed

This is not just an audit log. The provenance graph enables:
  - Trust chains: "I trust this because agent-A said it, and agent-A
    has been reliable"
  - Blame tracking: "This deployment failed because of a fact that
    was wrong, written by agent-X at time T"
  - Causal reasoning: "This decision was based on these 3 facts,
    and if fact #2 was wrong, the decision is invalidated"
  - Conflict resolution: "These two agents disagree, but agent-A's
    claim is derived from more primary sources"
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("dimensionalbase.reasoning.provenance")


class DerivationType(str, Enum):
    """How a piece of knowledge was derived."""
    ORIGINAL = "original"        # First-hand observation
    DERIVED = "derived"          # Computed from other entries
    CONFIRMED = "confirmed"      # Agent agreed with existing entry
    CONTRADICTED = "contradicted"  # Agent disagreed
    UPDATED = "updated"          # Same owner, new version
    SUMMARIZED = "summarized"    # Auto-generated summary
    MERGED = "merged"            # Composed from multiple entries


@dataclass
class ProvenanceNode:
    """A single node in the provenance DAG."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: str = ""
    value_hash: str = ""         # Hash of the value (for change detection)
    owner: str = ""
    timestamp: float = field(default_factory=time.time)
    derivation: DerivationType = DerivationType.ORIGINAL
    parent_ids: List[str] = field(default_factory=list)  # Edges in the DAG
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @property
    def is_root(self) -> bool:
        """Is this a root node (no parents)?"""
        return len(self.parent_ids) == 0


class ProvenanceTracker:
    """Tracks the full derivation history of all knowledge.

    The provenance graph is a DAG where:
      - Nodes are versions of knowledge entries
      - Edges point from derived entries to their sources
      - Root nodes are original observations
      - The graph can be traversed in both directions

    This enables causal reasoning, trust chains, and blame tracking.
    """

    def __init__(self, max_history: int = 50000):
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._path_to_nodes: Dict[str, List[str]] = {}  # path -> list of node IDs (versioned)
        self._children: Dict[str, Set[str]] = {}  # node_id -> set of child node_ids
        self._lock = threading.Lock()
        self._max_history = max_history

    def record_creation(
        self,
        path: str,
        owner: str,
        value_hash: str,
        derived_from: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """Record the creation of a new knowledge entry."""
        parent_ids = []
        derivation = DerivationType.ORIGINAL

        if derived_from:
            # Find the latest node for each source path
            for source_path in derived_from:
                latest = self._get_latest_node(source_path)
                if latest:
                    parent_ids.append(latest.id)
            if parent_ids:
                derivation = DerivationType.DERIVED

        node = ProvenanceNode(
            path=path,
            value_hash=value_hash,
            owner=owner,
            derivation=derivation,
            parent_ids=parent_ids,
            metadata=metadata or {},
            version=1,
        )

        with self._lock:
            self._add_node(node)

        return node

    def record_update(
        self,
        path: str,
        owner: str,
        value_hash: str,
        version: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """Record an update to an existing entry."""
        parent_ids = []
        latest = self._get_latest_node(path)
        if latest:
            parent_ids = [latest.id]

        node = ProvenanceNode(
            path=path,
            value_hash=value_hash,
            owner=owner,
            derivation=DerivationType.UPDATED,
            parent_ids=parent_ids,
            metadata=metadata or {},
            version=version,
        )

        with self._lock:
            self._add_node(node)

        return node

    def record_confirmation(
        self,
        path: str,
        confirming_agent: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProvenanceNode]:
        """Record that an agent confirmed an entry."""
        latest = self._get_latest_node(path)
        if not latest:
            return None

        node = ProvenanceNode(
            path=path,
            value_hash=latest.value_hash,
            owner=confirming_agent,
            derivation=DerivationType.CONFIRMED,
            parent_ids=[latest.id],
            metadata=metadata or {},
            version=latest.version,
        )

        with self._lock:
            self._add_node(node)

        return node

    def record_contradiction(
        self,
        path: str,
        contradicting_agent: str,
        new_value_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProvenanceNode]:
        """Record that an agent contradicted an entry."""
        latest = self._get_latest_node(path)
        if not latest:
            return None

        node = ProvenanceNode(
            path=path,
            value_hash=new_value_hash,
            owner=contradicting_agent,
            derivation=DerivationType.CONTRADICTED,
            parent_ids=[latest.id],
            metadata=metadata or {},
            version=latest.version + 1,
        )

        with self._lock:
            self._add_node(node)

        return node

    def record_merge(
        self,
        path: str,
        source_paths: List[str],
        owner: str,
        value_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceNode:
        """Record a merge of multiple entries into one."""
        parent_ids = []
        for sp in source_paths:
            latest = self._get_latest_node(sp)
            if latest:
                parent_ids.append(latest.id)

        node = ProvenanceNode(
            path=path,
            value_hash=value_hash,
            owner=owner,
            derivation=DerivationType.MERGED,
            parent_ids=parent_ids,
            metadata=metadata or {},
        )

        with self._lock:
            self._add_node(node)

        return node

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_lineage(self, path: str, max_depth: int = 20) -> List[ProvenanceNode]:
        """Get the full lineage of an entry (ancestors, newest first).

        Traces back through the DAG from the latest version to the roots.
        """
        latest = self._get_latest_node(path)
        if not latest:
            return []

        visited = set()
        lineage = []
        queue = [latest]

        while queue and len(lineage) < max_depth:
            node = queue.pop(0)
            if node.id in visited:
                continue
            visited.add(node.id)
            lineage.append(node)

            with self._lock:
                for parent_id in node.parent_ids:
                    if parent_id in self._nodes and parent_id not in visited:
                        queue.append(self._nodes[parent_id])

        return lineage

    def get_descendants(self, path: str) -> List[ProvenanceNode]:
        """Get all entries derived from this one."""
        latest = self._get_latest_node(path)
        if not latest:
            return []

        visited = set()
        descendants = []
        queue = [latest.id]

        with self._lock:
            while queue:
                node_id = queue.pop(0)
                if node_id in visited:
                    continue
                visited.add(node_id)

                children = self._children.get(node_id, set())
                for child_id in children:
                    if child_id in self._nodes:
                        descendants.append(self._nodes[child_id])
                        queue.append(child_id)

        return descendants

    def get_roots(self, path: str) -> List[ProvenanceNode]:
        """Find the original root sources for an entry."""
        lineage = self.get_lineage(path, max_depth=100)
        return [n for n in lineage if n.is_root]

    def get_history(self, path: str) -> List[ProvenanceNode]:
        """Get all versions of an entry, newest first."""
        with self._lock:
            node_ids = self._path_to_nodes.get(path, [])
            nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        return sorted(nodes, key=lambda n: n.timestamp, reverse=True)

    def get_causal_chain(
        self, path_a: str, path_b: str
    ) -> Optional[List[ProvenanceNode]]:
        """Find the causal chain connecting two entries, if any.

        Returns the shortest path through the DAG, or None if unconnected.
        """
        node_a = self._get_latest_node(path_a)
        node_b = self._get_latest_node(path_b)
        if not node_a or not node_b:
            return None

        # BFS from A looking for B
        visited = set()
        queue: List[Tuple[str, List[ProvenanceNode]]] = [(node_a.id, [node_a])]

        with self._lock:
            while queue:
                current_id, chain = queue.pop(0)
                if current_id == node_b.id:
                    return chain
                if current_id in visited:
                    continue
                visited.add(current_id)

                # Check both directions (parents and children)
                neighbors = set()
                node = self._nodes.get(current_id)
                if node:
                    neighbors.update(node.parent_ids)
                neighbors.update(self._children.get(current_id, set()))

                for neighbor_id in neighbors:
                    if neighbor_id not in visited and neighbor_id in self._nodes:
                        queue.append((neighbor_id, chain + [self._nodes[neighbor_id]]))

        return None

    def trust_depth(self, path: str) -> int:
        """How many unique agents have contributed to this entry's lineage."""
        lineage = self.get_lineage(path)
        return len(set(n.owner for n in lineage))

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance DAG for durable persistence."""
        with self._lock:
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "path": node.path,
                        "value_hash": node.value_hash,
                        "owner": node.owner,
                        "timestamp": node.timestamp,
                        "derivation": node.derivation.value,
                        "parent_ids": list(node.parent_ids),
                        "metadata": node.metadata,
                        "version": node.version,
                    }
                    for node in self._nodes.values()
                ]
            }

    def load_dict(self, payload: Optional[Dict[str, Any]]) -> None:
        """Restore provenance DAG from persisted data."""
        with self._lock:
            self._nodes = {}
            self._path_to_nodes = {}
            self._children = {}

            if not payload:
                return

            raw_nodes = payload.get("nodes", [])
            nodes: List[ProvenanceNode] = []
            for node_payload in raw_nodes:
                if not isinstance(node_payload, dict):
                    continue
                try:
                    node = ProvenanceNode(
                        id=str(node_payload.get("id", uuid.uuid4().hex[:12])),
                        path=str(node_payload.get("path", "")),
                        value_hash=str(node_payload.get("value_hash", "")),
                        owner=str(node_payload.get("owner", "")),
                        timestamp=float(node_payload.get("timestamp", time.time())),
                        derivation=DerivationType(node_payload.get("derivation", DerivationType.ORIGINAL.value)),
                        parent_ids=[str(parent_id) for parent_id in node_payload.get("parent_ids", [])],
                        metadata=node_payload.get("metadata", {}) or {},
                        version=int(node_payload.get("version", 1)),
                    )
                except Exception:
                    continue
                nodes.append(node)

            for node in sorted(nodes, key=lambda item: item.timestamp):
                self._add_node(node)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_node(self, node: ProvenanceNode):
        """Add a node to the DAG (must hold lock)."""
        self._nodes[node.id] = node

        if node.path not in self._path_to_nodes:
            self._path_to_nodes[node.path] = []
        self._path_to_nodes[node.path].append(node.id)

        # Update children index
        for parent_id in node.parent_ids:
            if parent_id not in self._children:
                self._children[parent_id] = set()
            self._children[parent_id].add(node.id)

        # Evict old nodes if over capacity
        if len(self._nodes) > self._max_history:
            self._evict_oldest()

    def _get_latest_node(self, path: str) -> Optional[ProvenanceNode]:
        """Get the most recent provenance node for a path."""
        with self._lock:
            node_ids = self._path_to_nodes.get(path, [])
            if not node_ids:
                return None
            # Latest by timestamp
            nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
            if not nodes:
                return None
            return max(nodes, key=lambda n: n.timestamp)

    def _evict_oldest(self):
        """Remove the oldest 10% of nodes."""
        all_nodes = sorted(self._nodes.values(), key=lambda n: n.timestamp)
        to_remove = all_nodes[:len(all_nodes) // 10]
        for node in to_remove:
            del self._nodes[node.id]
            if node.path in self._path_to_nodes:
                self._path_to_nodes[node.path] = [
                    nid for nid in self._path_to_nodes[node.path]
                    if nid != node.id
                ]
