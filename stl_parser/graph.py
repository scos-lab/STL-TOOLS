# -*- coding: utf-8 -*-
"""
STL Graph Construction

This module provides functionality to convert STL ParseResult objects
into graph structures using the networkx library. It allows for graph-based
analysis of STL statements.
"""

import networkx as nx
from typing import List, Dict, Any, Optional

from .models import ParseResult, Statement, Anchor
from .errors import STLGraphError, ErrorCode


class STLGraph:
    """
    Represents and analyzes STL statements as a directed graph.

    This class uses the networkx library to build a MultiDiGraph,
    allowing for multiple edges between the same two nodes, each
    representing a unique STL statement.
    """

    def __init__(self, parse_result: Optional[ParseResult] = None):
        """
        Initializes the STLGraph.

        Args:
            parse_result: An optional ParseResult object to build the graph from.
        """
        self.graph = nx.MultiDiGraph()
        if parse_result:
            self.build_graph(parse_result)

    def build_graph(self, parse_result: ParseResult) -> None:
        """
        Builds the graph from a ParseResult object.

        Each statement is converted into a directed edge. Nodes are anchor names.
        Modifiers are stored as edge attributes.

        Args:
            parse_result: The ParseResult object containing STL statements.
        """
        if not parse_result.is_valid:
            raise STLGraphError(
                code=ErrorCode.E300_GRAPH_CONSTRUCTION_FAILED,
                message="Cannot build graph from an invalid ParseResult."
            )

        for stmt in parse_result.statements:
            source_id = self._get_anchor_id(stmt.source)
            target_id = self._get_anchor_id(stmt.target)

            # Add nodes if they don't exist
            if not self.graph.has_node(source_id):
                self.graph.add_node(source_id, anchor=stmt.source.model_dump())
            if not self.graph.has_node(target_id):
                self.graph.add_node(target_id, anchor=stmt.target.model_dump())

            # Add edge with modifiers as attributes
            edge_attrs = stmt.modifiers.model_dump(exclude_unset=True) if stmt.modifiers else {}
            edge_attrs['statement_obj'] = stmt # Keep the original statement object
            
            self.graph.add_edge(source_id, target_id, **edge_attrs)

    def _get_anchor_id(self, anchor: Anchor) -> str:
        """
        Generates a unique string identifier for an anchor.
        Format: [Namespace:Name] or [Name]

        Args:
            anchor: The Anchor object.

        Returns:
            A unique string identifier for the anchor.
        """
        if anchor.namespace:
            return f"[{anchor.namespace}:{anchor.name}]"
        return f"[{anchor.name}]"

    def find_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """
        Finds all simple paths from a source node to a target node.

        Args:
            source_id: The identifier of the source anchor.
            target_id: The identifier of the target anchor.

        Returns:
            A list of paths, where each path is a list of anchor IDs.
        """
        if not self.graph.has_node(source_id):
            raise STLGraphError(code=ErrorCode.E302_INVALID_NODE, message=f"Source node '{source_id}' not in graph.")
        if not self.graph.has_node(target_id):
            raise STLGraphError(code=ErrorCode.E302_INVALID_NODE, message=f"Target node '{target_id}' not in graph.")
            
        return list(nx.all_simple_paths(self.graph, source=source_id, target=target_id))

    def find_cycles(self) -> List[List[str]]:
        """
        Finds all simple cycles in the graph.

        Returns:
            A list of cycles, where each cycle is a list of anchor IDs.
        """
        return list(nx.simple_cycles(self.graph))

    def get_node_degree(self, anchor_id: str) -> int:
        """
        Gets the total degree (in-degree + out-degree) of a node.

        Args:
            anchor_id: The identifier of the anchor.

        Returns:
            The total degree of the node.
        """
        if not self.graph.has_node(anchor_id):
            raise STLGraphError(code=ErrorCode.E302_INVALID_NODE, message=f"Node '{anchor_id}' not in graph.")
        return self.graph.degree(anchor_id)

    def get_node_centrality(self) -> Dict[str, float]:
        """
        Calculates the degree centrality for all nodes in the graph.

        Returns:
            A dictionary mapping anchor IDs to their degree centrality.
        """
        return nx.degree_centrality(self.graph)

    def get_subgraph(self, domain: str) -> 'STLGraph':
        """
        Creates a subgraph containing only statements from a specific domain.

        Args:
            domain: The domain to filter by (from the 'domain' modifier).

        Returns:
            A new STLGraph object representing the subgraph.
        """
        subgraph = STLGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('domain') == domain:
                if not subgraph.graph.has_node(u):
                    subgraph.graph.add_node(u, **self.graph.nodes[u])
                if not subgraph.graph.has_node(v):
                    subgraph.graph.add_node(v, **self.graph.nodes[v])
                subgraph.graph.add_edge(u, v, **data)
        
        return subgraph

    def detect_conflicts(self, functional_relations: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Detects semantic conflicts in the graph.
        A conflict is defined as a source node having multiple different targets 
        for the same functional relation.

        Args:
            functional_relations: Set of relation types considered functional (unique target).
                                Defaults to {"has_color", "is_a", "located_in", "defined_as"}.

        Returns:
            A list of conflict dictionaries.
        """
        target_relations = functional_relations or {"has_color", "is_a", "located_in", "defined_as"}
        conflicts = []

        for node in self.graph.nodes():
            relations = {}
            for _, target, data in self.graph.out_edges(node, data=True):
                # Determine relation type from modifiers or path type
                rel_type = data.get('rule') or data.get('path_type') or "generic"
                # Check for explicit 'relation' modifier if available
                if 'relation' in data:
                    rel_type = data['relation']

                if rel_type in target_relations:
                    if rel_type not in relations:
                        relations[rel_type] = []
                    
                    # Get target label/id
                    target_label = target
                    if 'anchor' in self.graph.nodes[target]:
                         target_label = self.graph.nodes[target]['anchor'].get('name', target)

                    relations[rel_type].append({
                        'target': target_label,
                        'confidence': data.get('confidence', 1.0),
                        'data': data
                    })

            # Check for conflicts (multiple unique targets for same functional relation)
            for rel_type, targets in relations.items():
                if len(targets) > 1:
                    unique_targets = set(t['target'] for t in targets)
                    if len(unique_targets) > 1:
                        # Get source label
                        source_label = node
                        if 'anchor' in self.graph.nodes[node]:
                            source_label = self.graph.nodes[node]['anchor'].get('name', node)

                        conflicts.append({
                            'source': source_label,
                            'relation': rel_type,
                            'targets': targets,
                            'tension_score': sum(t['confidence'] for t in targets)
                        })

        return conflicts

    def calculate_tension_metrics(self) -> Dict[str, Any]:
        """
        Calculates graph-wide semantic tension metrics.

        Returns:
            A dictionary containing conflict count, total tension, and average tension.
        """
        conflicts = self.detect_conflicts()
        total_tension = sum(c['tension_score'] for c in conflicts)
        return {
            "conflict_count": len(conflicts),
            "total_tension_score": total_tension,
            "avg_tension_per_conflict": total_tension / len(conflicts) if conflicts else 0.0
        }
        
    @property
    def summary(self) -> Dict[str, int]:
        """
        Provides a summary of the graph.

        Returns:
            A dictionary with node and edge counts.
        """
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    def extract_chains(self, min_length: int = 2) -> List[List[str]]:
        """
        Extracts all maximal directed chains from the graph.

        A chain is a maximal simple path: it starts at a node with in-degree 0
        (or a branch point) and ends at a node with out-degree 0 (or a merge
        point). Branches produce separate chains.

        Args:
            min_length: Minimum number of edges in a chain (default 2, i.e.
                        at least 3 nodes). Set to 1 to include single-edge
                        chains as well.

        Returns:
            A list of chains, where each chain is a list of anchor IDs
            (e.g. ``["[A]", "[B]", "[C]"]``).
        """
        if self.graph.number_of_nodes() == 0:
            return []

        # Collect source nodes (in-degree 0) — natural chain starts
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        # Collect sink nodes (out-degree 0) — natural chain ends
        sinks = set(n for n in self.graph.nodes() if self.graph.out_degree(n) == 0)

        # If no source nodes (pure cycles), use all nodes as potential starts
        if not sources:
            sources = list(self.graph.nodes())

        chains: List[List[str]] = []
        seen_chains: set = set()  # deduplicate by tuple

        for src in sources:
            if sinks:
                for sink in sinks:
                    if src == sink:
                        continue
                    try:
                        for path in nx.all_simple_paths(self.graph, src, sink):
                            key = tuple(path)
                            if len(path) - 1 >= min_length and key not in seen_chains:
                                seen_chains.add(key)
                                chains.append(path)
                    except nx.NetworkXError:
                        continue
            else:
                # Pure cycle graph — DFS from each node
                self._dfs_chains(src, [src], set(), chains, seen_chains, min_length)

        # Sort: longest first, then alphabetical
        chains.sort(key=lambda c: (-len(c), c))
        return chains

    def _dfs_chains(
        self,
        node: str,
        current_path: List[str],
        visited: set,
        chains: List[List[str]],
        seen_chains: set,
        min_length: int,
    ) -> None:
        """DFS helper for extracting chains in cyclic graphs."""
        visited.add(node)
        successors = [s for s in self.graph.successors(node) if s not in visited]

        if not successors:
            # End of path
            key = tuple(current_path)
            if len(current_path) - 1 >= min_length and key not in seen_chains:
                seen_chains.add(key)
                chains.append(list(current_path))
        else:
            for succ in successors:
                current_path.append(succ)
                self._dfs_chains(succ, current_path, visited, chains, seen_chains, min_length)
                current_path.pop()

        visited.discard(node)

    @staticmethod
    def format_chains(chains: List[List[str]]) -> str:
        """
        Formats extracted chains as human-readable text.

        Args:
            chains: Output of :meth:`extract_chains`.

        Returns:
            Formatted string, one chain per line.
        """
        if not chains:
            return "No chains found."
        lines = []
        for i, chain in enumerate(chains, 1):
            lines.append(f"Chain {i}: {' → '.join(chain)}")
        return "\n".join(lines)

    @staticmethod
    def from_parse_result(parse_result: ParseResult) -> 'STLGraph':
        """
        Factory method to create an STLGraph from a ParseResult.

        Args:
            parse_result: The ParseResult to convert.

        Returns:
            A new STLGraph instance.
        """
        return STLGraph(parse_result)

    @staticmethod
    def from_networkx(graph: nx.DiGraph) -> 'STLGraph':
        """
        Factory method to wrap an existing NetworkX directed graph.

        This allows external systems (e.g. STG) that already have a
        NetworkX graph to reuse STLGraph's analysis methods like
        :meth:`extract_chains`, :meth:`find_cycles`, etc.

        Node names are used as-is (no ``[bracket]`` wrapping).

        Args:
            graph: A NetworkX DiGraph or MultiDiGraph.

        Returns:
            A new STLGraph instance wrapping the given graph.
        """
        stl_graph = STLGraph()
        stl_graph.graph = graph
        return stl_graph
