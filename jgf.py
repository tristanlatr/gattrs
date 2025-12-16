"""
JSON Graph Format (JGF) 
https://jsongraphformat.info/
"""

from typing import List, Optional, Dict, Any

class Guard:
    """
    Various guards.
    """

    @staticmethod
    def assert_non_empty_string_parameter(name: str, value: Any) -> None:
        """
        Asserts that the value is a string and is not empty.
        """
        if not isinstance(value, str) or not value:
            # Using f-string for string interpolation
            raise ValueError(f'Parameter "{name}" has to be a non-empty string.')

    @staticmethod
    def assert_valid_metadata(metadata: Any) -> None:
        """
        Asserts that metadata is a non-empty dictionary.
        """
        # Checks if it is a dict and checks truthiness (empty dicts evaluate to False)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError('Metadata on a node has to be an object.')

    @staticmethod
    def assert_valid_metadata_or_null(metadata_or_null: Optional[Any]) -> None:
        """
        Asserts that metadata is valid if it is not None.
        """
        # check.assigned() matches "is not None" in Python
        if metadata_or_null is not None:
            Guard.assert_valid_metadata(metadata_or_null)

    @staticmethod
    def assert_valid_directed(directed: Any) -> None:
        """
        Asserts that the directed flag is a boolean.
        """
        if not isinstance(directed, bool):
            raise ValueError('Directed flag on an edge has to be boolean.')

class JgfEdge:
    """
    An edge object represents an edge between two nodes in a graph.
    In graph theory, edges are also called lines or links.
    """

    def __init__(self, source: str, target: str, relation: Optional[str] = None, 
                 label: Optional[str] = None, metadata: Optional[Dict] = None, 
                 directed: bool = True):
        """
        Constructor.
        
        Note: We assign directly to self.variable_name here to ensure
        the @setter validation logic (defined below) is triggered during initialization.
        """
        self.source = source
        self.target = target
        self.relation = relation
        self.label = label
        self.metadata = metadata
        self.directed = directed

    # Source Property
    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        Guard.assert_non_empty_string_parameter('source', value)
        self._source = value

    # Target Property
    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, value: str) -> None:
        Guard.assert_non_empty_string_parameter('target', value)
        self._target = value

    # Metadata Property
    @property
    def metadata(self) -> Optional[Dict]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Dict]) -> None:
        Guard.assert_valid_metadata_or_null(value)
        self._metadata = value

    # Directed Property
    @property
    def directed(self) -> bool:
        return self._directed

    @directed.setter
    def directed(self, value: bool) -> None:
        Guard.assert_valid_directed(value)
        self._directed = value

    def is_equal_to(self, edge: 'JgfEdge', compare_relation: bool = False) -> bool:
        """
        Determines whether this edge is equal to the passed edge.
        
        :param edge: The edge to compare to.
        :param compare_relation: Whether or not to compare the relation as well.
        """
        return (
            edge.source == self.source
            and edge.target == self.target
            and (not compare_relation or edge.relation == self.relation)
        )


class JgfNode:
    """
    A node object represents a node in a graph. 
    In graph theory, nodes are also called points or vertices.
    """

    def __init__(self, id: str, label: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Constructor.
        
        :param id: Primary key for the node, that is unique for the object type.
        :param label: A text display for the node.
        :param metadata: Metadata about the node.
        """
        self.id = id
        self.label = label
        
        # Assigning to self.metadata triggers the validation setter defined below
        self.metadata = metadata

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]) -> None:
        Guard.assert_valid_metadata_or_null(value)
        self._metadata = value

class JgfGraph:
    """
    A graph object represents the full graph and contains all nodes and edges 
    that the graph consists of.
    """

    def __init__(self, type: str = '', label: str = '', directed: bool = True, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Constructor.
        
        :param type: Graph classification.
        :param label: A text display for the graph.
        :param directed: Pass True for a directed graph, False for an undirected graph.
        :param metadata: Custom graph metadata.
        """
        self._nodes: List[JgfNode] = []
        self._edges: List[JgfEdge] = []

        self.type = type
        self.label = label
        self.directed = directed
        # Assigning to self.metadata triggers the validation setter defined below
        self.metadata = metadata

    # Metadata Property
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]) -> None:
        Guard.assert_valid_metadata_or_null(value)
        self._metadata = value

    @property
    def nodes(self) -> List[JgfNode]:
        """Returns all nodes."""
        return self._nodes

    @property
    def edges(self) -> List[JgfEdge]:
        """Returns all edges."""
        return self._edges

    def _find_node_by_id(self, node_id: str) -> JgfNode:
        """
        Finds a node by ID internally.
        :param node_id: Node to be found.
        :raises ValueError: If node does not exist.
        """
        # Python equivalent of _.find using a generator
        found_node = next((n for n in self._nodes if n.id == node_id), None)
        
        if not found_node:
            raise ValueError(f"A node does not exist with id = {node_id}")

        return found_node

    def _node_exists(self, node: JgfNode) -> bool:
        """
        Checks if a node object exists in the graph.
        """
        return self._node_exists_by_id(node.id)

    def _node_exists_by_id(self, node_id: str) -> bool:
        """
        Checks if a node ID exists in the graph.
        """
        found_node = next((n for n in self._nodes if n.id == node_id), None)
        return found_node is not None

    def add_node(self, node: JgfNode) -> None:
        """
        Adds a node to the graph.
        :param node: Node to be added.
        :raises ValueError: If the node already exists.
        """
        if self._node_exists(node):
            raise ValueError(f"A node already exists with id = {node.id}")

        self._nodes.append(node)

    def add_nodes(self, nodes: List[JgfNode]) -> None:
        """
        Adds multiple nodes to the graph.
        :param nodes: A collection of JgfNode objects to be added.
        """
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node: JgfNode) -> None:
        """
        Removes an existing node from the graph.
        :param node: Node to be removed.
        :raises ValueError: If the node does not exist.
        """
        if not self._node_exists(node):
            raise ValueError(f"A node does not exist with id = {node.id}")

        # Python equivalent of _.remove: rebuild list excluding the match
        self._nodes = [n for n in self._nodes if n.id != node.id]

    def get_node_by_id(self, node_id: str) -> JgfNode:
        """
        Get a node by a node ID.
        :param node_id: Unique node ID.
        """
        return self._find_node_by_id(node_id)

    def add_edge(self, edge: JgfEdge) -> None:
        """
        Adds an edge to the graph.
        :param edge: The edge to be added.
        """
        self._guard_against_non_existent_nodes(edge.source, edge.target)
        self._edges.append(edge)

    def _guard_against_non_existent_nodes(self, source: str, target: str) -> None:
        """
        Ensures both source and target nodes exist in the graph.
        """
        if not self._node_exists_by_id(source):
            raise ValueError(f"add_edge failed: source node isn't found in nodes. source = {source}")

        if not self._node_exists_by_id(target):
            raise ValueError(f"add_edge failed: target node isn't found in nodes. target = {target}")

    def add_edges(self, edges: List[JgfEdge]) -> None:
        """
        Adds multiple edges to the graph.
        :param edges: A collection of JgfEdge objects to be added.
        """
        for edge in edges:
            self.add_edge(edge)

    def remove_edge(self, edge: JgfEdge) -> None:
        """
        Removes existing edge from the graph.
        :param edge: Edge to be removed.
        """
        # Python equivalent of _.remove with custom equality check
        self._edges = [e for e in self._edges if not e.is_equal_to(edge, True)]

    def get_edges_by_nodes(self, source: str, target: str, relation: Optional[str] = None) -> List[JgfEdge]:
        """
        Get edges between source node and target node, with an optional edge relation.
        :param source: Source node ID.
        :param target: Target node ID.
        :param relation: If passed, only edges having this relation will be returned.
        """
        self._guard_against_non_existent_nodes(source, target)

        # Create a temporary edge for comparison
        temp_edge = JgfEdge(source, target, relation)

        # check.assigned(relation) is equivalent to 'relation is not None'
        compare_relation = relation is not None
        
        # Python equivalent of _.filter
        return [
            e for e in self._edges 
            if e.is_equal_to(temp_edge, compare_relation)
        ]

    @property
    def graph_dimensions(self) -> Dict[str, int]:
        """
        Returns the dimensions of the graph (count of nodes and edges).
        """
        return {
            'nodes': len(self._nodes),
            'edges': len(self._edges),
        }

class JgfMultiGraph:
    """
    Container for multiple Jgf graph objects in one object.
    """

    def __init__(self, type: str, label: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Constructor.
        
        :param type: Multi graph classification.
        :param label: A text display for the multi graph.
        :param metadata: Custom multi graph metadata.
        """
        self.type = type
        self.label = label
        
        # Initialize internal graphs list
        self._graphs: List[JgfGraph] = []

        # Note: In the JS constructor, it assigns directly to this._metadata, 
        # bypassing the setter validation (which strictly requires a non-empty object).
        # We mirror that behavior here to allow None during initialization.
        self._metadata = metadata

    def add_graph(self, graph: JgfGraph) -> None:
        """
        Adds a single graph.
        :param graph: Graph to be added.
        """
        self._graphs.append(graph)

    @property
    def graphs(self) -> List[JgfGraph]:
        """
        Returns all graphs.
        """
        return self._graphs

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        Returns the multi graph meta data.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Sets the multi graph meta data.
        """
        # The JS code uses Guard.assertValidMetadata here (not ...OrNull)
        # This implies that if you set metadata explicitly via property, 
        # it must be a valid, non-empty object.
        Guard.assert_valid_metadata(metadata)
        self._metadata = metadata

class JgfJsonDecorator:
    """
    Transforms graphs or multi graphs to json (dict) or vice versa.

    Note that this is just called decorator for semantic reasons and does not follow 
    and does not intend to follow the GoF decorator design pattern.
    """

    @staticmethod
    def from_json(json_data: Dict[str, Any]) -> JgfGraph | JgfMultiGraph:
        """
        Creates a Jgf graph or multi graph from JSON (dict).
        :param json_data: JSON to be transformed. This has to be according to the JGF.
        :raises ValueError: If json can not be transformed to a graph or multi graph.
        :returns: The created Jgf graph or multi graph object.
        """
        if not isinstance(json_data, dict):
            raise TypeError('json_data has to be a dict.')
        if 'graph' in json_data and json_data['graph'] is not None:
            return JgfJsonDecorator._graph_from_json(json_data['graph'])

        if 'graphs' in json_data and json_data['graphs'] is not None:
            # MultiGraph constructor might expect type/label/metadata, defaulting to None/empty if missing
            mg_type = json_data.get('type', '')
            mg_label = json_data.get('label', '')
            mg_metadata = json_data.get('metadata', None)
            
            graph = JgfMultiGraph(mg_type, mg_label, mg_metadata)
            
            for graph_json in json_data['graphs']:
                graph.add_graph(JgfJsonDecorator._graph_from_json(graph_json))

            return graph

        raise ValueError('Passed json has to have a "graph" or "graphs" property.')

    @staticmethod
    def _graph_from_json(graph_json: Dict[str, Any]) -> JgfGraph:
        """
        Creates a single JGF graph from JSON.
        """
        # Extract fields with safe defaults
        g_type = graph_json.get('type', '')
        g_label = graph_json.get('label', '')
        # Default to True as per JgfGraph constructor default
        g_directed = graph_json.get('directed', True) 
        g_metadata = graph_json.get('metadata', None)

        graph = JgfGraph(g_type, g_label, g_directed, g_metadata)

        nodes = graph_json.get('nodes', [])
        for node in nodes:
            # JgfNode constructor: id, label, metadata
            n_id = node.get('id')
            n_label = node.get('label')
            n_meta = node.get('metadata', None)
            graph.add_node(JgfNode(n_id, n_label, n_meta))

        edges = graph_json.get('edges', [])
        for edge in edges:
            # JgfEdge constructor: source, target, relation, label, metadata, directed
            e_source = edge.get('source')
            e_target = edge.get('target')
            e_rel = edge.get('relation', None)
            e_label = edge.get('label', None)
            e_meta = edge.get('metadata', None)
            e_directed = edge.get('directed', True)
            
            graph.add_edge(JgfEdge(e_source, e_target, e_rel, e_label, e_meta, e_directed))

        return graph

    @staticmethod
    def _guard_against_invalid_graph_object(graph: Any) -> None:
        if not isinstance(graph, JgfGraph) and not isinstance(graph, JgfMultiGraph):
            raise ValueError('JgfJsonDecorator can only decorate graphs or multi graphs.')

    @staticmethod
    def to_json(graph: JgfGraph | JgfMultiGraph) -> Dict[str, Any]:
        """
        Transforms either a graph or a multi graph object to a JSON (dict) representation as per the spec.
        :param graph: The graph to be transformed to JSON.
        :raises ValueError: If the passed graph or multi graph can not be transformed to JSON.
        :returns: A JSON representation of the passed graph or multi graph as according to the JGF.
        """
        JgfJsonDecorator._guard_against_invalid_graph_object(graph)

        is_single_graph = isinstance(graph, JgfGraph)

        all_graphs_json: Dict[str, Any] = {
            'graphs': [],
        }

        JgfJsonDecorator._transform_graphs_to_json(graph, all_graphs_json)

        if is_single_graph:
            # If it was a single graph, we unwrap the list and return { "graph": ... }
            # Accessing index 0 is safe because _transform_graphs_to_json pushed exactly one graph
            return JgfJsonDecorator._remove_null_values({'graph': all_graphs_json['graphs'][0]})

        # If MultiGraph
        all_graphs_json['type'] = graph.type
        all_graphs_json['label'] = graph.label
        all_graphs_json['metadata'] = graph.metadata

        return JgfJsonDecorator._remove_null_values(all_graphs_json)

    @staticmethod
    def _transform_graphs_to_json(graph: JgfGraph | JgfMultiGraph, all_graphs_json: Dict[str, Any]) -> None:
        normalized_graph = JgfJsonDecorator._normalize_to_multi_graph(graph)
        
        for single_graph in normalized_graph.graphs:
            single_graph_json = {
                'type': single_graph.type,
                'label': single_graph.label,
                'directed': single_graph.directed,
                'metadata': single_graph.metadata,
                'nodes': [],
                'edges': [],
            }

            JgfJsonDecorator._nodes_to_json(single_graph, single_graph_json)
            JgfJsonDecorator._edges_to_json(single_graph, single_graph_json)

            all_graphs_json['graphs'].append(single_graph_json)

    @staticmethod
    def _remove_null_values(data: Any) -> Any:
        """
        Recursively removes dictionary keys where the value is None.
        Equivalent to lodash's deep filter for null values.
        """
        if isinstance(data, dict):
            return {
                k: JgfJsonDecorator._remove_null_values(v) 
                for k, v in data.items() 
                if v is not None
            }
        elif isinstance(data, list):
            return [JgfJsonDecorator._remove_null_values(v) for v in data]
        else:
            return data

    @staticmethod
    def _edges_to_json(graph: JgfGraph, json_obj: Dict[str, Any]) -> None:
        for edge in graph.edges:
            json_obj['edges'].append({
                'source': edge.source,
                'target': edge.target,
                'relation': edge.relation,
                'label': edge.label,
                'metadata': edge.metadata,
                'directed': edge.directed,
            })

    @staticmethod
    def _nodes_to_json(graph: JgfGraph, json_obj: Dict[str, Any]) -> None:
        for node in graph.nodes:
            json_obj['nodes'].append({
                'id': node.id,
                'label': node.label,
                'metadata': node.metadata,
            })

    @staticmethod
    def _normalize_to_multi_graph(graph: JgfGraph | JgfMultiGraph) -> JgfMultiGraph:
        if isinstance(graph, JgfGraph):
            # Create a temporary multigraph holder
            normalized_graph = JgfMultiGraph(type="", label="")
            normalized_graph.add_graph(graph)
            return normalized_graph
        
        return graph
