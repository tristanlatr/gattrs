"""
A SUBSET OF JSON Graph Format (JGF) V2 https://jsongraphformat.info/

The schema is simplified and adapted for serializing Python objects, but still complies with the upstream JSON schema.
"""

# TODOS: 
# - Make the relation field required on edges

from typing import List, Optional, Dict, Any

_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "http://jsongraphformat.info/v2.1/json-graph-schema.json",
  "title": "JSON Graph Schema",
  "oneOf": [
    {
      "type": "object",
      "properties": {
        "graph": { "$ref": "#/definitions/graph" }
      },
      "additionalProperties": False,
      "required": [
        "graph"
      ]
    },
  ],
  "definitions": {
    "graph": {
      "oneOf": [
        {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "label": { "type": "string" },
            "directed": { "type": [ "boolean" ], "default": True },
            "type": { "type": "string" },
            "metadata": { "type": [ "object" ] },
            "nodes": {
              "type": "object",
              "additionalProperties": { "$ref": "#/definitions/node" }
            },
            "edges": {
              "type": [ "array" ],
              "items": { "$ref": "#/definitions/edge" }
            }
          }
        }
      ]
    },
    "node": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "label": { "type": "string" },
        "metadata": { "type": "object" }
      }
    },
    "edge": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "source": { "type": "string" },
        "target": { "type": "string" },
        "relation": { "type": "string" },
        "metadata": { "type": [ "object" ] }, 
      },
      "required": [ "source", "target", "relation" ]
    },
  }
}

class _Guard:
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
            _Guard.assert_valid_metadata(metadata_or_null)

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

    def __init__(self, source: str, target: str, 
                 relation: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Constructor.
        
        Note: We assign directly to self.variable_name here to ensure
        the @setter validation logic (defined below) is triggered during initialization.
        """
        self.source = source
        self.target = target
        self.relation = relation
        self.metadata = metadata

    # Source Property
    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        _Guard.assert_non_empty_string_parameter('source', value)
        self._source = value

    # Target Property
    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, value: str) -> None:
        _Guard.assert_non_empty_string_parameter('target', value)
        self._target = value
    
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]) -> None:
        _Guard.assert_valid_metadata_or_null(value)
        self._metadata = value

    def __eq__(self, edge: 'JgfEdge') -> bool:
        """
        Determines whether this edge is equal to the passed edge.
        
        :param edge: The edge to compare to.
        """
        if not isinstance(edge, JgfEdge):
            return NotImplemented
        return (
            edge.source == self.source
            and edge.target == self.target
            and edge.relation == self.relation
            and edge.metadata == self.metadata
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
        _Guard.assert_valid_metadata_or_null(value)
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
        self._nodes: Dict[str, JgfNode] = {}
        self._edges: List[JgfEdge] = []

        self.type = type
        self.label = label
        self.directed = directed
        self.metadata = metadata

    # Metadata Property
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]) -> None:
        _Guard.assert_valid_metadata_or_null(value)
        self._metadata = value

    @property
    def nodes(self) -> Dict[str, JgfNode]:
        """
        Returns all nodes as a list for compatibility/iteration.
        """
        return self._nodes

    @property
    def edges(self) -> List[JgfEdge]:
        """Returns all edges."""
        return self._edges
    
    # Directed Property
    @property
    def directed(self) -> bool:
        return self._directed

    @directed.setter
    def directed(self, value: bool) -> None:
        _Guard.assert_valid_directed(value)
        self._directed = value

    def _find_node_by_id(self, node_id: str) -> JgfNode:
        """
        Finds a node by ID internally.
        :param node_id: Node to be found.
        :raises ValueError: If node does not exist.
        """
        node = self._nodes.get(node_id)
        
        if not node:
            raise ValueError(f"A node does not exist with id = {node_id}")

        return node

    def _node_exists(self, node: JgfNode) -> bool:
        """
        Checks if a node object exists in the graph.
        """
        return self._node_exists_by_id(node.id)

    def _node_exists_by_id(self, node_id: str) -> bool:
        """
        Checks if a node ID exists in the graph.
        V2 Optimization: O(1) lookup.
        """
        return node_id in self._nodes

    def add_node(self, node: JgfNode) -> None:
        """
        Adds a node to the graph.
        :param node: Node to be added.
        :raises ValueError: If the node already exists.
        """
        if self._node_exists(node):
            raise ValueError(f"A node already exists with id = {node.id}")

        self._nodes[node.id] = node

    def add_nodes(self, nodes: List[JgfNode]) -> None:
        """
        Adds multiple nodes to the graph.
        :param nodes: A collection of JgfNode objects to be added.
        """
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node: JgfNode | str) -> None:
        """
        Removes an existing node from the graph.
        :param node: Node object or Node ID string to be removed.
        :raises ValueError: If the node does not exist.
        """
        node_id = node.id if hasattr(node, 'id') else node

        if not self._node_exists_by_id(node_id):
            raise ValueError(f"A node does not exist with id = {node_id}")

        del self._nodes[node_id]

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
        # Note: Ideally edges would be stored in a set or map for performance, 
        # but for V2 compatibility we kept it as a list for now as IDs on edges are optional.
        self._edges = [e for e in self._edges if e != edge]

    def get_edges_by_nodes(self, source: str, target: str, relation: Optional[str] = None) -> List[JgfEdge]:
        """
        Get edges between source node and target node, with an optional edge relation.
        :param source: Source node ID.
        :param target: Target node ID.
        :param relation: If passed, only edges having this relation will be returned.
        """
        self._guard_against_non_existent_nodes(source, target)
        if relation is not None:
            temp_edge = JgfEdge(source, target, relation)
            return [e for e in self._edges if e == temp_edge]
        return [
            e for e in self._edges if e.source == source and e.target == target
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

class Jgf:
    """
    Transforms graphs or multi graphs to json (dict) or vice versa.
    """

    @staticmethod
    def from_json(json_data: Dict[str, Any], validate:bool=False) -> JgfGraph:
        """
        Creates a Jgf graph or multi graph from JSON (dict).
        :param json_data: JSON to be transformed.
        :param validate: If True, validates against the schema. 
        :raises ValueError: If json can not be transformed to a graph or multi graph.
        :returns: The created Jgf graph or multi graph object.        """
        if not isinstance(json_data, dict):
            raise TypeError('json_data has to be a dict.')
        if validate:
            from jsonschema import validate as jsonschema_validate
            jsonschema_validate(instance=json_data, schema=_SCHEMA)
        if 'graph' in json_data and json_data['graph'] is not None:
            return Jgf._graph_from_json(json_data['graph'])
        raise ValueError('Passed json has to have a "graph" property.')

    @staticmethod
    def _graph_from_json(graph_json: Dict[str, Any]) -> JgfGraph:
        """
        Creates a single JGF graph from JSON.
        """
        graph = JgfGraph(
            type=graph_json.get('type', ''),
            label=graph_json.get('label', ''),
            directed=graph_json.get('directed', True),
            metadata=graph_json.get('metadata', None),
        )

        # --- 1. Parse Nodes (V2: Dict mapping ID -> Properties) ---
        raw_nodes = graph_json.get('nodes', {})
        if isinstance(raw_nodes, dict):
            for nid, n_data in raw_nodes.items():
                node = JgfNode(
                    id=nid,
                    label=n_data.get('label'),
                    metadata=n_data.get('metadata'),
                )
                graph.add_node(node)

        # --- 2. Parse Standard Edges ---
        for e_data in graph_json.get('edges', []):
            edge = JgfEdge(
                source=e_data['source'],
                target=e_data['target'],
                relation=e_data.get('relation'),
                metadata=e_data.get('metadata'),
            )
            graph.add_edge(edge)

        return graph


    @staticmethod
    def to_json(graph: JgfGraph , validate:bool=False) -> Dict[str, Any]:
        """
        Transforms either a graph or a multi graph object to a JSON (dict) representation as per the spec.
        :param graph: The graph to be transformed to JSON.
        :raises ValueError: If the passed graph or multi graph can not be transformed to JSON.
        :returns: A JSON representation of the passed graph or multi graph as according to the JGF.
        """
        if not isinstance(graph, JgfGraph):
            raise TypeError('expected graph to be either JgfGraph, got ' + str(type(graph)))

        all_graphs_json: Dict[str, Any] = {
            'graphs': [],
        }

        Jgf._transform_graphs_to_json(graph, all_graphs_json)

        # If it was a single graph, we unwrap the list and return { "graph": ... }
        # Accessing index 0 is safe because _transform_graphs_to_json pushed exactly one graph
        dat = Jgf._remove_null_values({'graph': all_graphs_json['graphs'][0]})

        if validate:
            from jsonschema import validate as jsonschema_validate
            jsonschema_validate(instance=dat, schema=_SCHEMA)
        
        return dat
    
    @staticmethod
    def _transform_graphs_to_json(graph: JgfGraph, all_graphs_json: Dict[str, Any]) -> None:

        single_graph_json = {
            'type': graph.type,
            'label': graph.label,
            'directed': graph.directed,
            'metadata': graph.metadata,
            'nodes': {},
            'edges': [],
        }

        Jgf._nodes_to_json(graph, single_graph_json)
        Jgf._edges_to_json(graph, single_graph_json)

        all_graphs_json['graphs'].append(single_graph_json)

    @staticmethod
    def _remove_null_values(data: Any, _key:object=object()) -> Any:
        if isinstance(data, dict):
            return {
                k: Jgf._remove_null_values(v, k)  if _key != 'metadata' else v
                for k, v in data.items() 
                # Metadata is special cased because it can actually contain None values
                if v is not None or _key == 'metadata'
            }
        elif isinstance(data, list):
            return [Jgf._remove_null_values(v) for v in data]
        else:
            return data

    @staticmethod
    def _edges_to_json(graph: JgfGraph, json_obj: Dict[str, Any]) -> None:
        # if not graph.edges:
        #     del json_obj['edges']
        #     return

        for edge in graph.edges:
            json_obj['edges'].append({
                'source': edge.source,
                'target': edge.target,
                'relation': edge.relation,
                'metadata': edge.metadata,
            })

    @staticmethod
    def _nodes_to_json(graph: JgfGraph, json_obj: Dict[str, Any]) -> None:
        for nid, node in graph.nodes.items():
            node_payload = {}
            if node.label:
                node_payload['label'] = node.label
            if node.metadata:
                node_payload['metadata'] = node.metadata
            json_obj['nodes'][nid] = node_payload