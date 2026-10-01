"""
A minimal GRAPH STORE.

Idea: store facts as  node --relation--> node.  Each edge remembers which
document it came from. Starting at one node, we can "hop" along edges to find
connected facts, even if they live in completely different documents.

We use networkx (an in-memory Python graph). Real projects use Neo4j or similar,
but "find a node, then walk its edges" is the same idea.
"""
import networkx as nx


class GraphStore:
    def __init__(self, triples):
        # Undirected so we can walk in both directions (team -> lead, lead -> team)
        self.graph = nx.Graph()
        for subject, relation, obj, source_doc in triples:
            # Remember the original direction so we can print the fact correctly
            self.graph.add_edge(subject, obj, subject=subject, relation=relation,
                                object=obj, source=source_doc)

    def find_entities(self, text: str) -> list[str]:
        """ENTITY LINKING: which graph nodes are mentioned in this text?
        (Simple case-insensitive string match. Real systems use an LLM or NER.)"""
        text_lower = text.lower()
        return [node for node in self.graph.nodes if node.lower() in text_lower]

    def expand(self, start_nodes: list[str], hops: int = 1):
        """Walk `hops` steps out from the start nodes.
        Returns the edges walked as (subject, relation, object, source_doc)."""
        visited_edges = []
        seen = set()
        frontier = set(start_nodes)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    edge_key = frozenset([node, neighbor])
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)
                    data = self.graph.edges[node, neighbor]
                    visited_edges.append(
                        (data["subject"], data["relation"], data["object"], data["source"]))
                    next_frontier.add(neighbor)
            frontier = next_frontier
        return visited_edges
